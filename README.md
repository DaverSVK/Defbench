# defectbench – porovnanie CNN, attention modulov, U-Net a vizuálnych transformerov na detekciu povrchových defektov (HNS)

Reprodukovateľný tréningový a evaluačný projekt pre tri datasety povrchových defektov (NEU-DET,
Magnetic Tile, GC10-DET) a 27 registrovaných modelov/variantov. Každý model beží vo **vlastnom
overenom prostredí** a zapisuje predikcie v **jednotnom formáte**. Všetky metriky počíta **jeden
spoločný evaluátor** z uložených predikcií.

> Stav k 2026-09-29. Tabuľka [Stav overenia](#6-stav-overenia-čo-bolo-skutočne-spustené) presne
> rozlišuje, čo bolo na tomto stroji **spustené**, čo je **implementované, ale nespustené** a čo je
> **nepodporované**, vrátane dôvodu. Výsledkové čísla sú v README uvedené len pre behy, ktoré reálne
> prebehli.

---

## Obsah
1. [Architektúra projektu](#1-architektúra-projektu)
2. [Overené implementácie a zdroje](#2-overené-implementácie-a-zdroje)
3. [Datasety – reálny audit](#3-datasety--reálny-audit)
4. [Matica model × dataset × úloha](#4-matica-model--dataset--úloha)
5. [Inštalácia a príkazy](#5-inštalácia-a-príkazy)
6. [Stav overenia (čo bolo skutočne spustené)](#6-stav-overenia-čo-bolo-skutočne-spustené)
7. [Metodika](#7-metodika)
8. [Evaluácia a definície metrík](#8-evaluácia-a-definície-metrík)
9. [Výstupy](#9-výstupy)
10. [Testy](#10-testy)
11. [Odchýlky od upstream, obmedzenia a známe problémy](#11-odchýlky-od-upstream-obmedzenia-a-známe-problémy)

---

## 1. Architektúra projektu

```
configs/
  datasets/{neu_det,magnetic_tile,gc10_det}.yaml   pravidlá parsovania, skupiny, split, kanály
  models/*.yaml                                    register modelov (zdroj, commit, licencia, váhy, stav)
  profiles/{default,gpu8gb}.yaml                   hardvérové profily (12–16 GB / 8 GB)
  environments.yaml                                izolované prostredia (venv / Docker)
  third_party.lock.yaml                            pripnuté commity externých repozitárov
defectbench/
  cli.py                     jednotné CLI (audit-data, prepare-data, train, evaluate, predict,
                             benchmark, run-matrix, summarize, smoke-test, task-matrix, list-models)
  data/  parsers.py          rekurzívne objavovanie súborov, VOC XML, maskové priečinky (bez pevných ciest)
         audit.py            audit (počty, rozlíšenia, neplatné boxy/masky, duplicity, existujúci split)
         split.py            skupinový, multilabelovo stratifikovaný split + manifest + hashe
         export.py           COCO (0-based), COCO s odvodenými inštanciami, YOLO
         geometry.py         konvencie súradníc, letterbox/resize inverzie, mask→box
         torch_datasets.py   synchronizované augmentácie (torchvision v2), kanálová politika
  models/ registry.py        register + matica model×dataset×úloha
          class_mapping.py   explicitné mapovanie class ID pre každý framework
          cbam.py            oficiálny CBAM modul + integrácia do FPN
          adapters/          torchvision_frcnn, segmentation (U-Net/AttU-Net/TransUNet/Swin-Unet),
                             ultralytics_adapter, rfdetr_adapter, detr_family (DEIMv2/D-FINE/RT-DETRv2),
                             detectron2_adapter (ViTDet, DINO/detrex), mmdet_adapter (Cascade + Swin-T)
  training/engine.py         spoločný tréningový engine (AMP, akumulácia, warmup, early stop, resume, OOM)
  runners/run_adapter.py     vstupný bod vo framework-špecifickom prostredí (subprocess / kontajner)
  pipeline.py, orchestrate.py orchestrácia, smoke-test, run-matrix
  formats.py                 jednotný formát predikcií (detekcia aj segmentácia) + validácia
  evaluation/                detection.py (COCO + pracovný bod), anomaly.py (AUROC/AUPRC/AUPRO),
                             segmentation.py (Dice/IoU…, seg→det), run_eval.py
  benchmark.py               latencia (batch 1, warmup, CUDA sync, medián/p95)
  reporting/                 grafy, vizualizácie GT vs. predikcie, agregácia a Markdown report
docker/                      detectron2.Dockerfile, mmdet.Dockerfile
scripts/                     setup_env.py, clone_third_party.py
tests/                       malé testy kritických častí
```

**Tok dát.** `prepare-data` spraví audit surových dát, zapíše `manifest.json` (iba platné obrázky,
kanonické boxy v súradniciach originálu) a `splits/default.json` (priradenie + SHA-256 každého
obrázka + hash splitu). Adaptér exportuje z manifestu formát, ktorý potrebuje jeho framework
(`exports/<split_hash>/…`, zdieľané všetkými modelmi). Po tréningu zapíše predikcie validačnej
a testovacej množiny v jednotnom formáte. `evaluate` (vždy v `core` prostredí) počíta všetky
metriky **len z týchto súborov**.

**Izolácia prostredí.** Detectron2, MMDetection, Ultralytics, RF-DETR a DETR repozitáre sa nikdy
nemiešajú. `core`, `ultralytics`, `rfdetr` a `detr_family` sú samostatné venv (uv).
`detectron2` a `mmdet` sú Docker kontajnery, pretože potrebujú kompilované CUDA operácie.
Orchestrátor spúšťa adaptéry ako subprocesy; zdieľa sa iba adresár behu.

---

## 2. Overené implementácie a zdroje

Pri každom modeli som pred implementáciou overil existenciu variantu, spôsob tréningu, dostupnosť
checkpointu, licenciu a podporované verzie priamo z oficiálneho repozitára alebo dokumentácie.
Commity sú HEAD predvolenej vetvy k 2026-09-29 (GitHub API). URL váh boli overené HTTP požiadavkou.

| # | Požiadavka | Registrovaný názov | Implementácia / zdroj | Verzia / commit | Licencia | Predtrénovanie |
|---|---|---|---|---|---|---|
| 1 | YOLO26 n/s | `yolo26n`, `yolo26s` | Ultralytics (YOLO26, jan. 2026, NMS-free) | ultralytics 8.4.165 | AGPL-3.0 | COCO, assets v8.4.0 |
| 2 | YOLOv10 n/s | `yolov10n`, `yolov10s` | Ultralytics (autori THU-MIG/yolov10) | ultralytics 8.4.165 | AGPL-3.0 | COCO, assets v8.4.0 |
| 3 | Faster R-CNN R50-FPN | `faster_rcnn_r50_fpn` | torchvision `fasterrcnn_resnet50_fpn` | torchvision 0.26.0 | BSD-3 | COCO_V1 (37.0 AP) |
| 4 | + CBAM | `faster_rcnn_r50_fpn_cbam` | + oficiálny CBAM (Jongchan/attention-module) | `459efad` | BSD-3 / MIT | rovnaké ako #3 |
| 5 | YOLOv12 n/s | `yolo12n`, `yolo12s` | Ultralytics YOLO12 (fork: sunsmarterjie/yolov12) | ultralytics 8.4.165 | AGPL-3.0 | COCO, assets v8.4.0 |
| 6 | ViTDet (menší backbone) | `vitdet_b_mask_rcnn_boxonly` | detectron2 `projects/ViTDet` (ViT-B = najmenší vydaný) | detectron2 `70f4543` | Apache-2.0 | COCO `model_final_61ccd1.pkl` (51.6 AP) |
| 7 | ViT-Adapter + DINO | `vit_adapter_dino` | **nepodporované** (pozri nižšie) | czczup/ViT-Adapter `fd64a51` | Apache-2.0 | – |
| 8 | RF-DETR menší | `rfdetr_nano`, `rfdetr_small` | roboflow/rf-detr (PyPI) | rfdetr 1.11.0 | Apache-2.0 | COCO (oficiálne) |
| 9 | DEIMv2 s ViT | `deimv2_dinov3_s` | Intellindust-AI-Lab/DEIMv2, variant **S** = ViT-Tiny destilovaný z DINOv3 | `1d2ca42` | DEIMv2 License (nekomerčné) | COCO (HF `Intellindust/DEIMv2_DINOv3_S_COCO`) |
| 10 | D-FINE menší | `dfine_s` | Peterande/D-FINE (recept `obj2custom`) | `956d170` | Apache-2.0 | Objects365→COCO |
| 11 | RT-DETRv2 menší | `rtdetrv2_s` | lyuwenyu/RT-DETR `rtdetrv2_pytorch` (R18-vd) | `29320b6` | Apache-2.0 | COCO (48.1 AP) |
| 12 | DINO + Swin-T | `dino_swin_t_4scale` | IDEA-Research/detrex `dino_swin_tiny_224_4scale_12ep` | `e244e6c` | Apache-2.0 | COCO (51.3 AP) |
| 13 | Cascade Mask R-CNN + Swin-T | `cascade_mask_rcnn_swin_t`, `cascade_rcnn_swin_t_boxonly` | MMDetection 3.3.0 komponenty v zložení oficiálneho Swin-OD configu | mmdet `cfd5d3a` | Apache-2.0 | oficiálny Swin-OD Cascade Mask R-CNN Swin-T 3x (50.4 AP) |
| 14 | U-Net | `unet_r34` (+ kontrola `unet_r34_scratch`) | segmentation_models_pytorch `Unet` | smp 0.5.0 | MIT | ImageNet ResNet-34 |
| 15 | Attention U-Net | `attention_unet` | MONAI `AttentionUnet` (Oktay et al.) | monai 1.6.1 | Apache-2.0 | **žiadne** (explicitne od nuly) |
| 16 | TransUNet | `transunet_r50_vit_b16` | Beckschen/TransUNet (oficiálny) | `02ef001` | Apache-2.0 | R50+ViT-B/16 ImageNet-21k |
| 17 | Swin-Unet | `swin_unet_t` | HuCaoFighting/Swin-Unet (oficiálny) | `f48f623` | bez LICENSE súboru | Swin-T ImageNet-1k |

Poznámky k overeniu:

- **DINO vs. DINOv2/DINOv3.** `dino_swin_t_4scale` je *detektor* DINO (Zhang et al., ICLR 2023).
  DINOv2/DINOv3 sú samoučiace reprezentačné modely; tu vystupujú len ako backbone (DEIMv2-S:
  ViT-Tiny destilovaný z DINOv3; RF-DETR: DINOv2-destilovaný backbone).
- **DINO + Swin-T.** Autorský repozitár IDEA-Research/DINO vydáva len R50 a Swin-L. Swin-T config
  aj COCO checkpoint vydáva tá istá skupina v detrex, preto sa použil detrex.
- **ViT-Adapter + DINO je nepodporované.** V oficiálnom repozitári `detection/` neobsahuje DINO.
  Jediné DINO configy (`wsdm2023/`) sú jazykovo podmienený GroundingDINO s `num_classes=1` pre úlohu
  vizuálneho groundingu (WSDM Cup 2023) s backbone Uni-Perceiver-B/L, teda nie multi-class
  detektor. Celý kód navyše vyžaduje mmdet 2.22 / mmcv-full 1.4.2 / torch 1.9 + CUDA 11.1, čo na
  sm_120 (RTX 50xx) nebeží. Neoficiálna kompozícia by bola nový, nepublikovaný model, preto sa
  **nenahrádza potichu**. Najbližšie overené alternatívy sú ViTDet-B a DINO-Swin-T, vedené pod
  vlastnými menami.
- **ViTDet.** Najmenší vydaný ViTDet backbone je ViT-B (ViT-S checkpoint neexistuje). Keďže
  datasety nemajú inštančné masky, masková vetva je vypnutá (`roi_heads.mask_in_features=None`)
  a model má názov `…_boxonly`.
- **Cascade Mask R-CNN + Swin-T.** Presná kombinácia je publikovaná v
  SwinTransformer/Swin-Transformer-Object-Detection (fork mmdet 2.11, mmcv-full 1.3, CUDA 11),
  čo na sm_120 nebeží. Architektúra je preto zostavená z komponentov MMDetection 3.3:
  Swin-T backbone, FPN [96, 192, 384, 768], 3× `ConvFCBBoxHead` 4conv1fc s GIoU (identické
  s oficiálnym configom) a AdamW 1e-4 / wd 0.05. Inicializuje sa **z oficiálneho COCO
  checkpointu**, pričom kľúče backbone konvertuje `mmdet.models.backbones.swin.swin_converter`.
- **YOLO12.** Ultralytics dokumentácia upozorňuje na možnú nestabilitu tréningu a vyššiu
  pamäťovú náročnosť. Použitá je Ultralytics implementácia, pretože zdieľa trénovací a
  evaluačný kód s YOLO26/YOLOv10 (kontrolované porovnanie).
- **DEIMv2 je registrovaný raz** (pokrýva aj kategóriu „detektor s ViT backbone“). To isté platí
  pre Swin-Unet: jeden záznam, hoci patrí do viacerých kategórií.

---

## 3. Datasety – reálny audit

Audit **prebehol na skutočných dátach** stiahnutých z uvedených Kaggle stránok (anonymne cez
`kagglehub`, bez prihlasovacích údajov). Úplné správy: `data/processed/<dataset>/audit/audit_report.{md,json}`.

| | NEU-DET | Magnetic Tile | GC10-DET |
|---|---|---|---|
| nájdené obrázky | 1800 | 1344 | 2312 súborov (2300 unikátnych) |
| platné obrázky | 1800 | 1340 | 2291 |
| triedy | 6 | 5 (+ normálne) | 10 |
| anotácie | 4189 boxov (VOC XML) | pixelové masky (sémantické); 462 **odvodených** boxov | 3562 boxov (VOC XML) |
| normálne vzorky | nie | áno, 952 (MT_Free) | nie |
| rozlíšenie | 200×200 | 1221 rôznych, 103–632 × 222–403 | 2048×1000 |
| kanály | RGB súbory s 3 identickými kanálmi (overené na všetkých 1800) | L | L |
| obrázky s >1 triedou | 123 | 0 | 475 |
| existujúci split | train/validation (1440/360) z Kaggle priečinkov | žiadny | žiadny |

Zistené problémy (všetko z auditu, nič nie je predpokladané):

- **NEU-DET**: obsahuje detekčné VOC anotácie (nie je to klasifikačná verzia). 1 pár pixelovo
  identických obrázkov (`patches_101` = `patches_105`), 3 duplicitné boxy, 81 objektov s príznakom
  `difficult`. Pole `<folder>` v XML obsahuje skratky (`cr`, …), čo je len informatívne.
- **Magnetic Tile**: masky nie sú čisto 0/255 (388 masiek má mäkké okraje s hodnotami 1–254),
  preto sa binarizujú pri hodnote ≥ 128. 4 obrázky z defektných priečinkov majú prázdnu masku a sú
  vylúčené (nemôžu byť ani negatívom). Kľúčové zistenie: obrázky sú **opakované zábery toho istého
  výrezu** v `exp1…exp6` (po sebe idúce `num`, rovnaké rozmery výrezu, pHash vzdialenosť 0–4).
  Náhodný split by preto spôsobil masívny leakage. Riešenie je popísané v časti
  [Rozdelenie dát](#72-rozdelenie-dát-a-zabránenie-leakage).
- **GC10-DET**: 6 obrázkov bez XML (vylúčené), 2 XML bez objektov (vylúčené, v defektnom priečinku
  nie sú dôveryhodné ako negatív), 1 objekt s neplatným menom `d` (obrázok vylúčený, inak by defekt
  ostal neoznačený), preklep `10_yaozhed` (131×) popri `10_yaozhe` (12×) mapovaný na
  `waist_folding`. 12 názvov súborov je v dvoch priečinkoch tried s pixelovo identickým obsahom
  a jedným XML, preto sú zlúčené na jeden záznam. Súbor `Defects Description.xlsx` určuje
  anglické názvy tried.

Duplicity: presné (hash bajtov aj dekódovaných pixelov) a podozrivo podobné (64-bit pHash,
Hammingova vzdialenosť ≤ 4): NEU 0 párov, MT 1322 párov (24 medzi triedami), GC10 2 páry.

---

## 4. Matica model × dataset × úloha

Úplná matica sa generuje z anotácií zistených auditom: `python -m defectbench task-matrix --output reports/task_matrix.json`.
Pravidlá:

| Situácia | Rozhodnutie |
|---|---|
| detektor × NEU / GC10 | `run`, pôvodné boxy |
| detektor × Magnetic Tile | `run`, boxy **odvodené** z masiek cez connected components (8-susednosť, bez filtra plochy); vo výsledkoch označené |
| segmentátor × Magnetic Tile | `run`, skutočné pixelové masky |
| segmentátor × NEU / GC10 | `not_applicable` – boxy sa nikdy nevypĺňajú do masiek. Segmentácia z boxov je povolená len ako samostatný, explicitne zapnutý weakly-supervised experiment (`allow_weak_box_supervision`, predvolene vypnuté) |
| Cascade **Mask** R-CNN × MT | `run` s maskovou loss na skutočných pixeloch, **inštancie odvodené** (CC sémantickej masky), označené |
| Cascade **Mask** R-CNN × NEU / GC10 | `not_applicable`; namiesto toho `cascade_rcnn_swin_t_boxonly` (bez mask head, bez mask loss) |
| image-level AUROC/AUPRC | len MT (má normálne obrázky); NEU/GC10 → N/A s dôvodom |
| pixel AUROC/AUPRC, AUPRO | len segmentátory na MT (skutočné masky + spojitá mapa) |
| ViT-Adapter + DINO | `unsupported` s dôvodom |

Normálne obrázky (MT_Free) zostávajú v detekcii ako negatívy s prázdnymi anotáciami a v segmentácii
ako obrázky s nulovou maskou.

---

## 5. Inštalácia a príkazy

### 5.1 Požiadavky
- Windows 10/11 alebo Linux, NVIDIA GPU s ovládačom pre CUDA 12.8 (RTX 50xx vyžaduje torch ≥ 2.7),
  [uv](https://github.com/astral-sh/uv) (odporúčané) alebo Python 3.11, git.
- Pre ViTDet, DINO-Swin-T a Cascade(-Mask) R-CNN Swin-T: Docker s NVIDIA runtime
  (Docker Desktop + WSL2 na Windows alebo nvidia-container-toolkit na Linuxe).

### 5.2 Prostredia
```bash
# 0) core prostredie (orchestrácia, dáta, evaluácia, Faster R-CNN, segmentátory)
uv venv envs/core --python 3.11
uv pip install -p envs/core/Scripts/python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p envs/core/Scripts/python.exe -r requirements/core.txt
#    (Linux: envs/core/bin/python)

# 1) externé repozitáre na pripnutých commitoch
envs/core/Scripts/python.exe scripts/clone_third_party.py

# 2) ďalšie izolované prostredia (venv)
envs/core/Scripts/python.exe scripts/setup_env.py ultralytics rfdetr detr_family

# 3) kontajnery (dlhý build: kompilácia detectron2/detrex resp. mmcv 2.1.0 pre CUDA 12.8)
envs/core/Scripts/python.exe scripts/setup_env.py detectron2 mmdet
```
Ďalej sa používa `python` = `envs/core/Scripts/python.exe` (Windows) alebo `envs/core/bin/python` (Linux).

### 5.3 Dáta
```bash
# stiahnutie z Kaggle (verejné datasety idú anonymne; ak Kaggle vyžaduje login, kagglehub si
# KAGGLE_USERNAME/KAGGLE_KEY alebo ~/.kaggle/kaggle.json načíta sám – projekt ich nečíta ani neloguje)
python -m defectbench prepare-data --dataset all --download
# alebo lokálne rozbalené archívy: data/raw/neu_det, data/raw/magnetic_tile, data/raw/gc10_det
python -m defectbench audit-data --dataset all        # samostatný audit (Markdown + JSON)
python -m defectbench prepare-data --dataset all      # audit + manifest + split (overí izoláciu)
```

### 5.4 Tréning, predikcia, evaluácia, latencia
```bash
# jeden beh = tréning + predikcia val/test + evaluácia (seed 17/42/73, profil 12–16 GB)
python -m defectbench train --dataset neu_det --model faster_rcnn_r50_fpn      --seed 42
python -m defectbench train --dataset neu_det --model faster_rcnn_r50_fpn_cbam --seed 42
python -m defectbench train --dataset neu_det --model rfdetr_small             --seed 42
python -m defectbench train --dataset magnetic_tile --model unet_r34           --seed 42
# 8 GB GPU: --profile gpu8gb (rovnaká efektívna dávka cez akumuláciu; výnimky v sekcii 7.4)
# zmena konfigurácie: --set train.epochs=50 --set train.batch_size=2 (zapíše sa do run_config.json)
# pokračovanie po prerušení (optimizer, scheduler, AMP scaler, early stopping, RNG):
python -m defectbench train --dataset neu_det --model faster_rcnn_r50_fpn --seed 42 --resume auto
#   --resume auto nájde posledný úplný stav frameworku (checkpoints/last.pt, Ultralytics weights/last.pt,
#   RF-DETR last.ckpt, DETR last.pth, detectron2 last_checkpoint, MMDetection epoch_*.pth); dá sa zadať aj cesta
# vlastný variant konfigurácie: --model cesta/k/variant.yaml (s `_base_: <model>`), beh sa pomenuje podľa `name`

# voliteľný, explicitne označený weakly-supervised experiment (segmentácia z vyplnených boxov, len NEU/GC10;
# hodnotí sa iba ako segmentation-derived detection voči skutočným boxom, pixelové metriky sú N/A):
python -m defectbench train --dataset neu_det --model unet_r34 --seed 42 --weak-box-supervision

python -m defectbench predict   --dataset neu_det --model faster_rcnn_r50_fpn --seed 42 [--checkpoint …]
python -m defectbench evaluate  --dataset neu_det --model faster_rcnn_r50_fpn --seed 42
python -m defectbench benchmark --dataset neu_det --model faster_rcnn_r50_fpn --seed 42 --n-images 200

# celá matica (preskočí neplatné kombinácie a nenainštalované prostredia, dokončené behy nechá tak)
python -m defectbench run-matrix --seeds 17,42,73 [--models …] [--datasets …] [--dry-run]
python -m defectbench summarize --output reports/generated     # results.csv/json + report.md + grafy

# rýchla kontrola funkčnosti (malá vzorka: načítanie, forward, backward, checkpoint, inferencia, evaluácia)
python -m defectbench smoke-test --dataset neu_det
python -m defectbench smoke-test --dataset magnetic_tile --models unet_r34,transunet_r50_vit_b16
python -m defectbench smoke-test --dataset neu_det --device cpu --models faster_rcnn_r50_fpn   # CPU smoke
python -m pytest -q                                            # unit testy
```

---

## 6. Stav overenia (čo bolo skutočne spustené)

Hardvér vývoja: **NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB** (sm_120), driver 572.97, Windows 11,
torch 2.11.0+cu128. Tento stroj je pod predvoleným profilom (12–16 GB), preto behy nižšie používajú
`--profile gpu8gb`.

**Legenda:** ✅ spustené tu · 🟡 implementované, na tomto stroji nespustené · ⛔ nepodporované.

| Model | Prostredie | Smoke test (NEU alebo MT) | Plný tréning tu |
|---|---|---|---|
| faster_rcnn_r50_fpn | core | ✅ NEU, MT | pozri 6.1 |
| faster_rcnn_r50_fpn_cbam | core | ✅ NEU | pozri 6.1 |
| unet_r34 | core | ✅ MT | ✅ MT seed 42 |
| unet_r34_scratch, attention_unet, transunet_r50_vit_b16, swin_unet_t | core | ✅ MT | – |
| rfdetr_nano | rfdetr | ✅ NEU | pozri 6.1 |
| rfdetr_small | rfdetr | rovnaký kód ako nano | – |
| yolo26n, yolov10n, yolo12n | ultralytics | ✅ NEU | – |
| yolo26s, yolov10s, yolo12s | ultralytics | rovnaký kód ako *n (iné váhy) | – |
| dfine_s, rtdetrv2_s, deimv2_dinov3_s | detr_family | ✅ NEU | – |
| vitdet_b_mask_rcnn_boxonly, dino_swin_t_4scale | detectron2 (Docker) | 🟡 pozri 6.2 | – |
| cascade_mask_rcnn_swin_t, cascade_rcnn_swin_t_boxonly | mmdet (Docker) | 🟡 pozri 6.2 | – |
| vit_adapter_dino | – | ⛔ | ⛔ |

Smoke test = tréning na 8 obrázkoch (1 epocha, niekoľko iterácií), uloženie checkpointu, inferencia
val/test s validáciou jednotného formátu (súradnice v rámci originálu, rozsah labelov) a úplná
evaluácia. Protokoly sú v `reports/smoke/*.json`. **Nie sú to experimentálne výsledky.**

### 6.1 Výsledky reálnych behov
Uvedené sú len behy, ktoré skutočne dobehli. Jeden seed = žiadna smerodajná odchýlka.

<!-- RESULTS-START -->
*Doplní sa po dobehnutí behov (pozri `reports/generated/report.md`).*
<!-- RESULTS-END -->

### 6.2 Docker prostredia
Dockerfiles pre `detectron2` (detectron2 + detrex, torch 2.11 cu128) a `mmdet` (mmcv 2.1.0 zo
zdrojov, torch 2.7.1 cu128) sú priložené a adaptéry sú úplne implementované. Stav buildu na tomto
stroji je uvedený v [časti 11](#11-odchýlky-od-upstream-obmedzenia-a-známe-problémy). Kým build
a smoke test neprebehnú, tieto adaptéry **nie sú označené za funkčné**.

---

## 7. Metodika

### 7.1 Konvencie súradníc a tried
- Boxy sú `[x1, y1, x2, y2]` v spojitých súradniciach hrán pixelov **originálneho** obrázka;
  box pokrývajúci stĺpce `c0..c1` má `x2 = c1 + 1`. VOC (1-based, inkluzívne) → `x1 = xmin − 1, x2 = xmax`.
- Kanonický label = index v `classes` datasetu. Posun každého frameworku je v
  `defectbench/models/class_mapping.py` (torchvision +1 kvôli pozadiu, COCO GT json +1, ostatné 0).
  DETR rodina dostáva COCO s 0-based `category_id` a `remap_mscoco_category=False`. RF-DETR:
  flat kategórie (label = poradie podľa ID); adaptér overí `class_names` a zahodí no-object slot
  s indexom K, ktorý samotné rfdetr označuje ako `__background__`.
- Každý adaptér vracia predikcie v súradniciach originálu (torchvision/detectron2/mmdet postprocess,
  Ultralytics `scale_boxes`, DETR `orig_size`, RF-DETR `predict`). Validátor formátu odmietne boxy
  mimo originálu, labely mimo rozsahu aj chýbajúce obrázky.

### 7.2 Rozdelenie dát a zabránenie leakage
- **Oficiálne rozdelenie** neexistuje pre MT ani GC10. NEU má v Kaggle archíve priečinky
  train/validation (1440/360). Vytvoril ich ten, kto dataset nahral na Kaggle, nie autori datasetu,
  a chýba test, preto sa nepovažujú za dôveryhodné oficiálne rozdelenie. Predvolený je nový split
  70/15/15. Voľba `split.strategy: provided` zachová Kaggle priečinky (validation → test,
  validačná množina sa vyreže z train).
- **Skupiny (union-find)**: rovnaký zdroj/sekvencia + pixelovo identické obrázky + podozrivé
  takmer-duplicity (pHash ≤ 4) patria do jedného splitu.
  - GC10: `img_<kamera>_<id>_<snímka>` → `<id>` je zdroj (zvitok/akvizícia), odvodené z pomenovania
    (predpoklad je zdokumentovaný). 465 skupín, najväčšia má 202 obrázkov.
  - MT: `exp<E>_num_<N>`, kde `N` je globálne počítadlo zberu. Nová skupina začína pri medzere
    `N` > 300 (naprieč triedami, pretože aj normálne výrezy z toho istého záberu idú spolu).
    176 skupín po pridaní pHash hrán.
  - NEU: žiadny identifikátor zdroja, iba duplicity.
- **Stratifikácia**: multilabelová iteratívna stratifikácia (Sechidis et al., 2011) nad skupinami.
  Vektor skupiny tvoria počty obrázkov s každou triedou (obrázky s viacerými triedami sa rátajú
  pre každú) plus počet normálnych obrázkov.
- Split sa robí **pred** augmentáciami. Seed splitu (20260929) je nezávislý od tréningových seedov
  17/42/73, takže všetky modely aj seedy používajú **ten istý manifest**. Verifikácia: 0 skupín
  cez viac splitov, 0 duplicít, 0 párov takmer-duplicít medzi splitmi. `prepare-data` odmietne
  potichu prepísať existujúci split s iným hashom.

| dataset | split hash | train / val / test (obrázky) | realizované pomery |
|---|---|---|---|
| neu_det | `39464da676e8b3b4` | 1224 / 288 / 288 | 0.68 / 0.16 / 0.16 |
| magnetic_tile | `e1a102671ee203ae` | 936 / 202 / 202 (normálne 668/143/141) | 0.70 / 0.15 / 0.15 |
| gc10_det | `9152b058f5f257da` | 1692 / 301 / 298 | 0.74 / 0.13 / 0.13 |

GC10 sa od 70/15/15 odchyľuje kvôli veľkým zdrojovým skupinám (202, 137, 137 obrázkov). Integrita
skupín má prednosť pred presným pomerom.

### 7.3 Reprodukovateľnosť
Každý beh zapíše `run_config.json` (úplná konfigurácia datasetu a modelu po profiloch a
`--set`, hash manifestu a splitu, kanálová politika, zdroj predtrénovania) a `environment.json`
(verzie balíkov, git commit projektu a každého repozitára v `third_party/`, GPU, driver, CUDA,
cuDNN). Premenné prostredia sa nelogujú, okrem úzkeho zoznamu (napr. `CUDA_VISIBLE_DEVICES`).
SHA-256 stiahnutých váh sa ukladá do `weights/registry.json`. Prahy a hyperparametre sa vyberajú
**iba na validačnej množine**. Test sa použije až v evaluácii a výber modelu (best checkpoint,
early stopping, pracovný bod, binarizačný prah, parametre seg→det) ho nikdy nevidí.

### 7.4 Tréning
- **Transfer learning** je predvolený všade, kde existujú vhodné váhy. Tréning od nuly musí byť
  explicitný (`pretrained.from_scratch_explicit: true`). Týka sa to len `attention_unet` (bez
  verejných 2D váh) a kontrolného `unet_r34_scratch`.
- **Spoločný engine** (`training/engine.py`, pre in-process adaptéry): CUDA + AMP (fp16 s
  GradScaler), akumulácia gradientu, warmup + cosine/multistep/poly (per iterácia), voliteľné
  zmrazenie backbone na N epoch (`train.freeze_backbone_epochs`: ResNet body pri Faster R-CNN,
  enkóder pri smp U-Net, transformer pri TransUNet, Swin enkóder pri Swin-Unet), early stopping
  podľa validačnej metriky, `best.pt` a `last.pt`,
  **skutočné resume** (model, optimizer, scheduler, GradScaler, počítadlá, stav early stoppingu,
  RNG, história), logovanie train aj val loss, čas a max. GPU pamäť. Resume sa overil aj naživo:
  U-Net beh na MT havaroval na prechodnej chybe cuDNN pri štarte Docker/WSL2, pokračoval z
  `last.pt` od epochy 63 s obnoveným stavom early stoppingu a dobehol.
- **Externé frameworky** používajú svoj oficiálny trénovací cyklus a projekt k nemu pridáva, čo
  chýba. Ultralytics a RF-DETR majú vlastný early stopping, EMA, AMP a úplné resume (`last.pt`,
  `last.ckpt`). U RF-DETR sa zapína `compute_val_loss`. Pri DEIMv2/D-FINE/RT-DETRv2 adaptér pridá
  early stopping a val loss obalením upstream `evaluate` bez zmeny repozitára; resume ide cez
  upstream `-r last.pth` (model, EMA, optimizer, scheduler, scaler). Detectron2/detrex dostáva
  hooky pre val loss, best checkpoint a early stopping; MMDetection `EarlyStoppingHook`,
  `save_best` a `AmpOptimWrapper`.
- **Výber modelu (best checkpoint, early stopping)** prebieha vždy na validácii. Detektory používajú
  COCO AP@[.50:.95] (framework-native evaluátor alebo náš). Segmentátory používajú **pixelový AP
  defektnej mapy bez prahu** (histogram s 1000 binmi pri sieťovom rozlíšení). Dice@0.5 sa pri
  výbere ukázal ako veľmi šumivý (0.0–0.77 medzi susednými epochami), preto sa len loguje.
- **Predvolené nastavenia podľa rodiny** (nie jeden optimizer pre všetko):
  Faster R-CNN SGD 0.01 @ ef. batch 8, multistep. smp U-Net/AttU-Net AdamW 3e-4, cosine.
  TransUNet SGD 0.01 @ 24, poly (oficiálny trainer, lineárne škálované). Swin-Unet SGD 0.05 @ 24,
  poly. YOLO Ultralytics defaulty (`optimizer=auto`, mosaic, close_mosaic, nbs 64). RF-DETR
  AdamW 1e-4 / encoder 1.5e-4 @ ef. 16. DETR rodina: oficiálny recept príslušného repozitára.
  ViTDet AdamW s layer decay 0.7. DINO AdamW 1e-4 (backbone ×0.1) s clip 0.1. Cascade Swin AdamW
  1e-4, wd 0.05.
- **Hardvérový profil.** Predvolený je **jedna GPU s 12–16 GB VRAM** (`profiles/default.yaml`).
  `--profile gpu8gb` mení dávku za akumuláciu pri **rovnakej efektívnej dávke** (Faster R-CNN,
  RF-DETR, Ultralytics cez `nbs`, Cascade Swin). **Výnimka:** upstream enginy DEIMv2/D-FINE/
  RT-DETRv2 nemajú akumuláciu, preto `gpu8gb` polovičnú dávku kompenzuje lineárnym škálovaním lr
  (pravidlo z D-FINE README). Ide o zmenu receptu a zapíše sa do `framework/batch_lr_scaling.json`.
  Náročné modely majú `heavy: true`: TransUNet, ViTDet-B, DINO-Swin-T, Cascade Swin-T.
- **OOM**: engine zachytí `torch.OutOfMemoryError`, zapíše `oom.json` a vypíše konkrétne nápravy
  (polovičná dávka + dvojnásobná akumulácia, profil `gpu8gb`, AMP, zmenšenie vstupu s upozornením,
  že mení metodiku). **Nič nemení sám.**
- **Augmentácie** sú synchrónne pre obrázok, boxy aj masky (torchvision v2). Pre in-process modely
  sa používajú len zmeny zachovávajúce jemné defekty: horizontálny/vertikálny flip a jas/kontrast
  ±0.2, bez náhodných výrezov, mozaiky a rozmazania. Externé rodiny používajú svoje odporúčané
  recepty (zdokumentované v configoch).
- **Kanály a normalizácia.** Všetky datasety sú šedotónové (NEU má RGB súbory s identickými
  kanálmi). Obrázok sa načíta jednokanálovo a **zreplikuje do RGB** (`replicate_gray_to_rgb`,
  zapísané v každom behu). Normalizácia zodpovedá predtrénovaniu: ImageNet mean/std, pre
  TransUNet (Google ViT) [−1, 1], pri torchvision/DETR/Ultralytics ju robí framework.
- **Vstupné rozlíšenie** sa ukladá do výsledkov. Faster R-CNN: min 800 / max 1333 (torchvision).
  YOLO: 640 letterbox. DETR rodina: 640. RF-DETR: natívne 384/512. ViTDet: LSJ 1024.
  Segmentátory: 224×224 (natívne pre TransUNet/Swin-Unet, spoločné pre všetky); predikcie sa
  bilineárne vracajú na originál.

---

## 8. Evaluácia a definície metrík

### 8.1 Bounding-box detekcia (spoločný evaluátor)
- Jednotný formát `defectbench.det.v1`: `image_id, boxes_xyxy, scores, labels`. Spoločný exportný
  prah je **0.001** a limit **100 detekcií na obrázok** pre všetky detektory. AP sa tak počíta
  z poradia skóre cez celý rozsah prahov, nie z binarizovaných detekcií.
- **pycocotools**: AP@[.50:.05:.95], AP50, AP75, AP po triedach (a AP50 po triedach), AR@1, AR@10,
  AR@100. Trieda bez GT v splite → `None` s dôvodom; split bez GT → AP `None`, nie 0.
- **Pracovný bod**: triedne uvedomelé párovanie jeden-na-jedného pri IoU ≥ 0.50. Predikcie idú
  zostupne podľa skóre a každá sa spáruje s nespárovaným GT tej istej triedy s najvyšším IoU.
  Druhá predikcia na ten istý objekt je **FP (duplicate)**. Prah skóre = argmax F1 na validácii
  (pri zhode vyšší prah), zmrazený pre test. Reportujú sa precision, recall a F1 pri tomto bode
  a typy FP: duplicate, localization (0.1 ≤ IoU < 0.5), classification a background.
- **COCO AR vs. recall v pracovnom bode**: AR@k priemeruje recall cez IoU 0.50–0.95, berie top-k
  detekcií a **nemá prah skóre**. Recall@wp je jedno IoU (0.50) a len detekcie nad zmrazeným
  prahom. Nie sú zameniteľné.

### 8.2 Image-level AUROC/AUPRC (normálny vs. defektný obrázok)
- Len ak split obsahuje pozitívne aj negatívne obrázky (MT). Inak `None` + dôvod a **žiadne umelé
  negatívy**.
- Skóre obrázka je stanovené pred testovaním: detektor = max. skóre defektu (0, ak nemá žiadnu
  detekciu). Segmentátor = priemer 100 najvyšších pixelov defektnej mapy (`topk_mean`, k = 100).

### 8.3 Pixel-level AUROC/AUPRC a AUPRO
- Len pri skutočných GT maskách a spojitej mape (segmentátory na MT). Z boxov sa **nikdy**
  nerobí pixelová mapa.
- Binárna úloha defekt vs. pozadie. Multiclass mapa = 1 − P(pozadie).
  **Pooled** metrika cez všetky pixely splitu, nie priemer per-image hodnôt.
  Obrázky bez defektu prispievajú len negatívnymi pixelmi.
- **AUPRO**: oblasti = 8-súvislé komponenty GT masky každého obrázka. PRO(t) = priemer cez všetky
  oblasti z |oblasť ∩ {skóre ≥ t}| / |oblasť|; FPR(t) = podiel všetkých GT-negatívnych pixelov so
  skóre ≥ t. Presná krivka pri každom unikátnom skóre, lichobežníková integrácia na
  **FPR ∈ [0, 0.30]**, lineárna interpolácia na hranici a **normalizácia deliteľom 0.30**
  (dokonalá mapa = 1). Krivka PRO–FPR sa ukladá. Implementácia je overená testami
  (dokonalá/invertovaná mapa, váhovanie oblastí, porovnanie s brute force, invariancia
  k monotónnej transformácii).
- **AUPRC konvencia (všade rovnaká)**: primárne **average precision** (stupňovitá suma,
  `sklearn.average_precision_score`, bez interpolácie). Sekundárne `auprc_trapezoid`
  (lichobežník pod PR krivkou) je uložené zvlášť. Ani jedno nie je COCO AP.

### 8.4 Segmentácia
- Binarizačný prah na validácii (maximum pooled Dice, krok 0.001), zmrazený pre test.
- Pooled Dice, IoU, pixel precision a recall (primárne), per-image priemery a výsledky po triedach
  (pixelová trieda = trieda priečinka MT).
- Prázdne masky: GT aj predikcia prázdna → Dice/IoU = 1; práve jedna prázdna → 0. Precision
  pri prázdnej predikcii a recall pri prázdnom GT sa vynechávajú a ich počet sa reportuje.
- **Segmentation-derived detection**: connected components binarizovanej mapy → boxy, skóre oblasti
  (max alebo priemer), label = argmax priemerných pravdepodobností tried. Prah, min. plocha a
  pravidlo skóre sa volia mriežkou na validácii (COCO AP) a výsledok je vo výsledkoch označený ako
  „segmentation-derived detection“.

### 8.5 Latencia
Batch 1, 20 zahrievacích iterácií, `torch.cuda.synchronize()` okolo každej fázy, prvých 200
testovacích obrázkov. Reportuje sa medián a p95 pre **čas modelu** (`model_fn`, rozsah opísaný pri
každom adaptéri) a **end-to-end** čas (načítanie + predspracovanie + model + postprocessing do
jednotného formátu). **FPS = 1000 / medián latencie pri batch 1.** Ukladá sa aj hardvér,
presnosť (fp32, ak nie je uvedené inak) a backend (PyTorch eager).

---

## 9. Výstupy

Každý beh (`runs/<dataset>/<model>/<task>/seed<seed>/`) obsahuje `run_config.json`,
`environment.json`, `params.json`, `train_log.csv`, `train_summary.json` (čas, peak GPU pamäť,
best epoch, early stopping), checkpointy, `predictions/{val,test}_det.json` alebo
`predictions/{val,test}_seg/` (pre nezávislé prepočítanie), `eval/metrics.json` (všetky metriky
aj dôvody nedostupných), `benchmark/latency.json` a `figures/` (PR/ROC, PRO–FPR, tréningové
krivky, AP po triedach, vizualizácie GT vs. predikcie s FP podľa typu a FN).

`summarize` vytvorí `reports/generated/`: `results.csv`, `results.json`,
`results_aggregated.json` a `report.md` (oddelené tabuľky bbox detekcie, segmentácie,
image-level detekcie anomálií a seg-derived detekcie **po datasetoch**, priemer ± std cez seedy,
výsledky po triedach, nedostupné metriky s dôvodom) plus graf presnosť vs. latencia.

---

## 10. Testy

`python -m pytest -q` (spustené na tomto stroji, všetky prešli):

| Súbor | Čo overuje |
|---|---|
| `test_geometry.py` | VOC → pixel-edge boxy, xywh/cxcywh, inverzia letterboxu a stretch resize, orezanie, IoU, mask→box (plochy, 4- vs. 8-susednosť) |
| `test_class_mapping.py` | round-trip mapovania class ID pre všetky frameworky, posun torchvision, kategórie COCO GT |
| `test_split.py` | skupiny, pixelové duplicity a takmer-duplicity nikdy cez splity, reprodukovateľnosť, pomery, prítomnosť tried, režim `provided` |
| `test_detection_evaluator.py` | dokonalá predikcia (AP = 1), zlá trieda (AP = 0), duplicity = FP, lokalizačné/background FP, prázdne predikcie aj prázdne GT, chýbajúca trieda = None, AP z poradia skóre, prah z validácie |
| `test_anomaly_metrics.py` | AUROC/AUPRC = None pri jednej triede, známe hodnoty AUROC/AP, AUPRO dokonalá/invertovaná mapa, váhovanie oblastí, zhoda s brute force, interpolácia na hranici, invariancia |
| `test_formats_and_seg_metrics.py` | validátor formátu (neobrátený resize, chýbajúce obrázky, zlé labely), top-k, pravidlá prázdnych masiek, výber prahu, metriky po triedach, seg→det |
| `test_cbam_and_resume.py` | CBAM variant pridá iba CBAM parametre (P2–P5) a pri rovnakom seede má zhodnú inicializáciu zdieľaných váh; P6 z CBAM-rafinovaného P5; resume = neprerušený tréning |

Smoke testy (`defectbench smoke-test`) sú integračné testy s reálnymi dátami a reálnymi
frameworkami. Ktoré prebehli, je uvedené v [časti 6](#6-stav-overenia-čo-bolo-skutočne-spustené).

---

## 11. Odchýlky od upstream, obmedzenia a známe problémy

- **torch 2.11.0+cu128 všade.** RTX 50xx (sm_120) potrebuje torch ≥ 2.7. DEIMv2 upstream pinuje
  torch 2.5.1, čo je odchýlka overená smoke testom. Kontajner `mmdet` používa torch 2.7.1.
- **DEIMv2 × torchvision 0.26**: niektoré transformácie DEIMv2 definujú len `_transform`.
  D-FINE a RT-DETRv2 (rovnaká línia kódu) majú upstream alias `transform → _transform`.
  Ten istý alias sa pridáva za behu (`detr_family._torchvision_v2_shim`). Na Windows ho kvôli
  spawn DataLoader workerom inštaluje `runners/shims/sitecustomize.py`, ktorý je aktívny len
  s `DEFECTBENCH_TV_SHIM=1` v detr_family subprocese. Repozitár sa nemení.
- **TransUNet na Windows**: oficiálny `load_from` skladá kľúče npz cez `os.path.join` (spätné
  lomky). Adaptér podáva pohľad s normalizovanými kľúčmi a upstream kód sa nemení.
- **RF-DETR**: predvolený eval backend `vernier` importuje pyarrow, ktorého DLL na vývojovom stroji
  blokuje Windows Application Control. Použitý je `faster_coco_eval`, ktorý balík dokumentuje ako
  metricky identický.
- **Swin-Unet** trénuje so spoločnou loss benchmarku 0.5 CE + 0.5 Dice, oficiálny trainer používa
  0.4/0.6. Repozitár nemá LICENSE súbor, preto ho používajte len na výskum.
- **DEIMv2 licencia** je nekomerčná (výskum a výučba).
- **DETR rodina @ gpu8gb** mení recept (polovičná dávka a lineárne lr), pozri 7.4.
- **Peak GPU pamäť** je maximum cez všetky relácie behu (uložené v checkpointe, resume ho zohľadní).
  Pri externých frameworkoch sa meria v procese adaptéra (`torch.cuda.max_memory_allocated`).
- **Docker behy** (ViTDet, DINO-Swin-T, Cascade Swin-T) sú na tomto stroji v stave
  „implementované“, kým kontajnery a ich smoke testy neprebehnú (pozri 6.2).
