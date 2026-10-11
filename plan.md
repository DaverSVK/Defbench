Si skúsený výskumný inžinier v oblasti PyTorch, detekcie objektov, segmentácie a priemyselnej vizuálnej kontroly.

Vytvor kompletný, spustiteľný Python projekt na trénovanie a vyhodnocovanie modelov pre detekciu povrchových defektov. Projekt slúži na výskumné porovnanie CNN, attention modulov, U-Net a vizuálnych transformerov v rámci predmetu HNS.

Nechcem iba návrh, pseudokód ani tréningové funkcie s TODO. Potrebujem konkrétne súbory, konfigurácie, inštalačné postupy a príkazy. Používaj existujúce oficiálne implementácie a predtrénované váhy. Nevytváraj zjednodušenú vlastnú sieť a neoznačuj ju názvom publikovaného modelu.

**1. Datasety**

Podporuj tieto tri datasety:

- NEU Surface Defect Database:
  https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data
- Magnetic Tile Defect Dataset:
  https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects
- GC10-DET – Metallic Surface Defect Detection:
  https://www.kaggle.com/datasets/alex000kim/gc10det/data

Trénuj na každom datasete samostatne. Nespájaj automaticky ich triedy ani obrázky.

Najprv over skutočnú štruktúru stiahnutých dát a anotácií. NEU musí obsahovať detekčné anotácie, nestačí klasifikačná verzia. Nepredpokladaj pevné názvy vnorených adresárov.

Implementuj audit dát, ktorý vypíše:

- počet obrázkov, tried a anotácií,
- počet obrázkov a objektov jednotlivých tried,
- rozlíšenia obrázkov,
- dostupnosť bounding boxov, masiek a normálnych vzoriek,
- chýbajúce alebo poškodené súbory,
- neplatné boxy a masky,
- presné duplicity a podozrivé podobné obrázky,
- existujúce rozdelenie train/validation/test.

Ak dáta nie sú dostupné v pracovnom prostredí, implementuj auditovací príkaz a uveď, že reálny audit ešte neprebehol. Nevymýšľaj výsledky.

Podporuj lokálne rozbalené datasety a voliteľné stiahnutie cez Kaggle. Prihlasovacie údaje nesmú byť súčasťou kódu ani logov.

**2. Modely**

Vytvor register modelov s jednoznačnými názvami a samostatnými adaptérmi pre tieto architektúry:

Detektory:

1. YOLO26, varianty n/s, ak existujú v overenej oficiálnej implementácii.
2. YOLOv10, varianty n/s, ako samostatná alternatíva.
3. Faster R-CNN + ResNet-50-FPN.
4. Faster R-CNN + ResNet-50-FPN + CBAM.
5. YOLOv12, varianty n/s.
6. ViTDet s dostupným menším predtrénovaným backbone.
7. ViT-Adapter + DINO.
8. RF-DETR, menší oficiálne podporovaný variant.
9. DEIMv2, vrátane explicitne zvoleného variantu s ViT backbone.
10. D-FINE, menší variant.
11. RT-DETRv2, menší variant.
12. DINO + Swin-T, ak je táto konfigurácia podporovaná.
13. Cascade Mask R-CNN + Swin-T, podľa dostupnosti vhodných anotácií.

Segmentačné modely:

14. U-Net.
15. Attention U-Net.
16. TransUNet.
17. Swin-Unet.

DEIMv2 a Swin-Unet implementuj iba raz, aj keď patria do viacerých koncepčných kategórií.

Rozlišuj detektor DINO od reprezentačných modelov DINOv2/DINOv3.

Pri CBAM zachovaj rovnaký základ ako pri referenčnom Faster R-CNN. Presne zdokumentuj, do ktorých FPN úrovní je CBAM vložený. Referenčný model a jeho CBAM variant musia umožniť kontrolované porovnanie.

Pred implementáciou over z oficiálnej dokumentácie alebo repozitárov:

- existenciu konkrétneho variantu,
- správny spôsob vytvorenia a trénovania modelu,
- dostupnosť checkpointov,
- licenciu,
- podporované verzie Python/PyTorch/CUDA a ostatných knižníc.

Nevymýšľaj API, názvy checkpointov ani kompatibilitu balíkov. Každý model spoj s konkrétnym zdrojom a verziou alebo commitom. Ak požadovaná kombinácia nie je dostupná, označ ju ako nepodporovanú a zdôvodni to. Nenahrádzaj ju potichu iným modelom.

**3. Správne priradenie úloh k dátam**

Vygeneruj maticu model × dataset × úloha podľa zistených anotácií.

- Bounding box anotácie používaj na detekciu.
- Pixelové masky používaj na segmentáciu.
- Z masiek možno odvodiť boxy cez connected components, ale označ ich ako odvodené anotácie a zdokumentuj pravidlá.
- Pri semantic maskách nepovažuj connected components automaticky za skutočné inštančné anotácie.
- Nevyplňuj bounding boxy a neprezentuj ich ako skutočné pixelové masky.
- Segmentáciu trénovanú z boxov povoľ iba ako samostatný, explicitne označený weakly supervised experiment.
- Cascade Mask R-CNN netrénuj so štandardnou maskovou loss, ak chýbajú vhodné masky. Prípadný box-only režim musí byť technicky podporovaný a jasne pomenovaný.
- Normálne obrázky zachovaj ako negatívne príklady s prázdnymi detekčnými anotáciami alebo nulovou maskou.

Nepokúšaj sa nasilu trénovať všetky modely na všetkých datasetoch, ak anotácie príslušnú úlohu nepodporujú.

**4. Rozdelenie dát a reprodukovateľnosť**

Ak existuje dôveryhodné oficiálne rozdelenie, zachovaj ho a zdokumentuj jeho pôvod. Inak vytvor reproducibilný split 70 % train / 15 % validation / 15 % test.

Zabráň data leakage:

- obrázky toho istého výrobku, sekvencie alebo zdroja patria do jedného splitu, ak sú tieto identifikátory dostupné,
- duplicity nesmú prechádzať medzi splitmi,
- rozdelenie rob pred augmentáciami a vytváraním výrezov,
- stratifikáciu prispôsob obrázkom s viacerými triedami.

Ulož manifest so splitom a kontrolnými hashmi. Všetky modely na danom datasete musia používať rovnaký manifest.

Podporuj seedy 17, 42 a 73. Predvolený split zostáva medzi tréningovými seedmi rovnaký.

Zaznamenaj konfiguráciu, verzie závislostí, git commity, GPU, predtrénovanie a checkpointy. Hyperparametre a prahy vyberaj iba na validačnej množine. Test nesmie ovplyvňovať výber modelu.

**5. Tréning**

Použi transfer learning, ak existujú vhodné váhy. Tréning od nuly musí byť explicitná voľba.

Implementuj:

- CUDA a AMP, ak sú podporované,
- CPU smoke test, kde je technicky možný,
- gradient accumulation,
- nastaviteľný batch size, počet epoch, learning rate a weight decay,
- scheduler, warmup a prípadné zmrazenie backbone,
- early stopping podľa validačnej metriky,
- uloženie best a last checkpointu,
- skutočné resume vrátane optimizeru, scheduleru a AMP stavu,
- logovanie tréningovej a validačnej loss,
- meranie času a maximálnej GPU pamäte.

Použi vhodné predvolené nastavenia pre jednotlivé rodiny modelov. Nevnucuj všetkým rovnaký optimizer len kvôli formálnej jednotnosti.

Predvoleným profilom nech je jedna GPU s približne 12–16 GB VRAM; tento predpoklad zdokumentuj. Náročné modely označ a poskytni menšie konfigurácie, ak existujú. Pri OOM vypíš konkrétnu nápravu, nemeň potichu metodiku experimentu.

Augmentácie musia synchronizovane meniť obrázok aj anotácie. Zachovaj jemné defekty. Normalizáciu a počet kanálov prispôsob predtrénovanému modelu; pri grayscale vstupoch zaznamenaj prípadné kopírovanie kanála do RGB.

**6. Spoločné vyhodnotenie detekcie**

Zjednoť výstup všetkých detektorov na:

image_id, boxes_xyxy, scores, labels.

Pred evaluáciou transformuj boxy späť do pôvodných súradníc obrázka. Over mapovanie class ID, odstránenie paddingu a reverziu resize/letterbox.

Použi spoločný COCO-kompatibilný evaluator, napríklad pycocotools, a reportuj:

- AP50,
- AP75,
- AP@[0.50:0.05:0.95],
- AP jednotlivých tried,
- AR@1, AR@10 a AR@100,
- recall pri IoU = 0.50 a validačne zvolenom confidence thresholde,
- precision a F1 pri rovnakom pracovnom bode.

Pri pracovnom bode definuj class-aware one-to-one matching predikcií a ground truth. Duplicitné detekcie musia byť false positives.

AP počítaj z confidence skóre cez rozsah prahov, nie iba z binarizovaných výsledkov pri jednom prahu. Použi dostatočne nízky exportný confidence threshold a spoločný limit počtu detekcií.

Zdokumentuj rozdiel medzi COCO AR a recall pri konkrétnom pracovnom bode.

**7. AUROC, AUPRC a AUPRO**

Tieto metriky nepočítaj automaticky pre každú kombináciu. Rozlišuj:

Image-level AUROC a AUPRC:

- Úloha: normálny obrázok verzus obrázok s defektom.
- Potrebné sú pozitívne aj negatívne obrázky.
- Pri detektoroch možno použiť maximum confidence skóre defektu; bez predikcií skóre 0.
- Pri segmentátoroch použi vopred definovanú agregáciu mapy, napríklad maximum alebo priemer top-k pixelov.
- Pravidlo agregácie nastav pred testovaním.
- Ak split obsahuje iba jednu triedu, vráť null/N/A s vysvetlením. Nevytváraj umelé negatívne vzorky.

Pixel-level AUROC a AUPRC:

- Počítaj ich iba pri skutočných ground-truth maskách a kontinuálnej skórovacej mape.
- Pre binárnu úlohu definuj defect-vs-background, pri viacerých triedach zdokumentuj agregáciu.
- Presne definuj, či ide o pooled pixel metric alebo priemer per-image metrík.
- Zdokumentuj správanie pri obrázkoch bez defektu alebo pri chýbajúcej triede.
- Z boxov nevytváraj implicitne pixelovú anomaly mapu. Ak taký experiment pridáš, označ ho oddelene ako odvodenú heuristiku.

AUPRO:

- Počítaj iba pri skutočných maskách a kontinuálnych mapách.
- Použi connected ground-truth defect regions a per-region overlap.
- Predvolený integračný rozsah nastav na FPR ∈ [0, 0.30].
- Uveď interpoláciu hranice a normalizáciu plochy rozsahom FPR.
- Ulož PRO–FPR krivku.
- Použi overenú implementáciu alebo svoju implementáciu over na malých syntetických príkladoch.
- Ak ju nemožno korektne určiť, reportuj N/A s dôvodom.

Pri AUPRC explicitne rozlišuj average precision od trapezoidálnej integrácie PR krivky. Zvolenú konvenciu používaj konzistentne a zaznamenaj v reporte. Nezamieňaj tieto hodnoty s detekčným COCO AP.

**8. Vyhodnotenie segmentácie**

Okrem pixelových AUROC/AUPRC a AUPRO reportuj:

- Dice,
- IoU,
- pixel precision,
- pixel recall,
- výsledky jednotlivých tried, ak sú dostupné.

Prah binarizácie vyber na validácii a zmraz pred testom. Definuj správanie metrík pri prázdnej predikovanej aj ground-truth maske.

Voliteľne vytvor z predikovaných masiek boxy cez connected components a vyhodnoť ich detekčným evaluatorom. Prah, filtrovanie malých oblastí a confidence oblasti stanov na validácii. Výsledky označ „segmentation-derived detection“.

**9. Férové porovnanie a výstupy**

Pre každé spustenie ulož:

- dataset, model, úlohu a seed,
- split hash a úplnú konfiguráciu,
- zdroj predtrénovania,
- vstupné rozlíšenie,
- počet parametrov,
- čas tréningu a peak GPU memory,
- všetky relevantné metriky,
- dôvody nepoužiteľných metrík,
- predikcie pre nezávislé prepočítanie metrík.

Reportuj priemer a smerodajnú odchýlku cez tréningové seedy. Pri jedinom seede nevymýšľaj variabilitu.

Oddeľ tabuľky bbox detekcie, segmentácie a image-level detekcie anomálií. Nemiešaj rôzne datasety do jedného priemeru.

Latenciu meraj po warmupe, s CUDA synchronizáciou, pri batch size 1. Uveď medián a p95, hardvér, precision a backend. Oddeľ čas modelu od end-to-end času vrátane preprocessingu a postprocessingu. Pri FPS jasne definuj spôsob výpočtu.

Exportuj:

- results.csv a results.json,
- súhrnnú Markdown správu,
- PR a ROC krivky, kde sú definované,
- PRO–FPR krivky,
- tréningové krivky,
- výsledky po triedach,
- vizualizácie ground truth a predikcií,
- false positives, false negatives a chybné lokalizácie,
- porovnanie presnosti a latencie.

Nevyrábaj výsledkové hodnoty, ak tréning alebo evaluácia neprebehli.

**10. Architektúra projektu a použitie**

Projekt rozdeľ aspoň na:

- konfigurácie datasetov a modelov,
- audit a konverziu dát,
- tvorbu splitov,
- modelové adaptéry,
- tréning,
- inferenciu,
- spoločné evaluátory,
- vizualizáciu a agregáciu výsledkov,
- malé testy kritických častí,
- README.

Poskytni jednotné CLI s operáciami:

- audit-data,
- prepare-data,
- train,
- evaluate,
- predict,
- benchmark,
- run-matrix,
- summarize,
- smoke-test.

CLI musí umožniť výber datasetu, modelu, konfigurácie, seedu, zariadenia a checkpointu.

Pre nezlučiteľné frameworky použi samostatné prostredia alebo kontajnery. Nenúť Detectron2, MMDetection, Ultralytics a jednotlivé DETR repozitáre do jedného neovereného prostredia. Spoločný orchestrátor môže spúšťať adaptéry ako subprocessy; evaluácia má používať jednotný formát predikcií.

**11. Overenie funkčnosti**

Pridaj malé, zmysluplné testy pre:

- konverziu boxov a súradníc,
- mapovanie class ID,
- mask-to-box konverziu,
- izoláciu splitov,
- dokonalú a chybnú predikciu v evaluátore,
- prázdne anotácie a prázdne predikcie,
- nedostupné AUROC pri jednej triede,
- správnosť AUPRO na syntetických mapách.

Smoke test má na malej vzorke overiť načítanie dát, forward, backward, uloženie checkpointu, inferenciu a evaluáciu. Jasne odlíš testy, ktoré si vykonal, od testov, ktoré iba poskytuješ.

**12. Poradie implementácie a forma odpovede**

Najprv priprav fungujúcu hlavnú vetvu:

1. audit a spoločné dátové formáty,
2. Faster R-CNN baseline,
3. CBAM variant,
4. jeden overený moderný DETR detektor,
5. U-Net na dátach so skutočnými maskami,
6. spoločná evaluácia a report.

Potom doplň ostatné požadované adaptéry. Rozsah projektu nezužuj bez vysvetlenia. Ak niektorú integráciu blokujú nedostupné závislosti, checkpointy alebo hardvér, označ presný stav a dôvod.

Začni stručným prehľadom architektúry projektu a overených implementácií. Potom vytvor súbory projektu a uveď presné príkazy od prípravy prostredia po vyhodnotenie.

Ak máš prístup k súborovému systému, vytváraj súbory priamo. Ak ho nemáš, poskytni úplný obsah súborov po logických častiach. Pri limite odpovede uveď dokončené a zostávajúce súbory; nedokončené adaptéry neoznačuj za funkčné.

Nezastavuj sa pri všeobecnom pláne. Cieľom je použiteľný tréningový a evaluačný projekt, ktorého výsledky bude možné obhájiť vo výskumnej práci.