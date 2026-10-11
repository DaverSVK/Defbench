"""PDF report: metric comparison, training curves (loss, val mAP / Dice, precision/recall where logged)
and strengths/weaknesses of the architecture families.

    envs/core/bin/python -m defectbench summarize --output reports/generated
    envs/core/bin/python scripts/make_pdf_report.py [--seed 42] [--out reports/defectbench_report.pdf]

Only runs with finished evaluation (reports/generated/results.csv) are used. Runs listed in EXCLUDE are
left out with the reason printed in the report.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"neu_det": "NEU-DET", "magnetic_tile": "Magnetic Tile", "gc10_det": "GC10-DET"}
EXCLUDE = {"dfine_s": "D-FINE: tréning konverguje, ale validačná/inferenčná cesta vracia nezmyselné predikcie "
                      "(AP ≈ 0 už od 1. epochy) – chyba integrácie, rieši sa; čísla nie sú platné."}

# architecture families = categories of the study
FAMILIES = {
    "CNN": ["yolo26n", "yolo26s", "yolov10n", "yolov10s", "faster_rcnn_r50_fpn"],
    "CNN + attention": ["yolo12n", "yolo12s", "faster_rcnn_r50_fpn_cbam"],
    "DETR rodina": ["rfdetr_nano", "rfdetr_small", "deimv2_dinov3_s", "rtdetrv2_s", "dfine_s"],
    "ViT / Swin backbone": ["vitdet_b_mask_rcnn_boxonly", "dino_swin_t_4scale", "cascade_rcnn_swin_t_boxonly",
                            "cascade_mask_rcnn_swin_t"],
    "U-Net a hybridy": ["unet_r34", "unet_r34_scratch", "attention_unet", "transunet_r50_vit_b16", "swin_unet_t"],
}
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}
DISPLAY = {
    "yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "yolov10n": "YOLOv10n", "yolov10s": "YOLOv10s",
    "yolo12n": "YOLOv12n", "yolo12s": "YOLOv12s", "faster_rcnn_r50_fpn": "Faster R-CNN R50-FPN",
    "faster_rcnn_r50_fpn_cbam": "Faster R-CNN R50-FPN + CBAM", "rfdetr_nano": "RF-DETR-N",
    "rfdetr_small": "RF-DETR-S", "deimv2_dinov3_s": "DEIMv2-S (DINOv3)", "rtdetrv2_s": "RT-DETRv2-S",
    "dfine_s": "D-FINE-S", "vitdet_b_mask_rcnn_boxonly": "ViTDet-B", "dino_swin_t_4scale": "DINO + Swin-T",
    "cascade_rcnn_swin_t_boxonly": "Cascade R-CNN + Swin-T", "cascade_mask_rcnn_swin_t": "Cascade Mask R-CNN + Swin-T",
    "unet_r34": "U-Net R34 (ImageNet)", "unet_r34_scratch": "U-Net R34 (od nuly)", "attention_unet": "Attention U-Net",
    "transunet_r50_vit_b16": "TransUNet R50-ViT-B/16", "swin_unet_t": "Swin-Unet-T",
}

# reference categorical palette (light), fixed order; adjacent-pair validated for lines (slots 1-8)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# scatter (all pairs in view): only slots 1-3 validate all-pairs -> 4th family = neutral ink + own marker
FAMILY_COLOR = {"CNN": SERIES[0], "CNN + attention": SERIES[1], "DETR rodina": SERIES[2],
                "ViT / Swin backbone": "#52514e"}
FAMILY_MARKER = {"CNN": "o", "CNN + attention": "s", "DETR rodina": "^", "ViT / Swin backbone": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BAR3 = SERIES[:3]  # AP50 / AP75 / AP50:95

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.6, "figure.dpi": 100,
})


def dn(m):
    return DISPLAY.get(m, m)


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


# ----------------------------------------------------------------------------------------------- data
def load_results(seed):
    d = pd.read_csv(ROOT / "reports" / "generated" / "results.csv")
    d = d[d.seed == seed].copy()
    excluded = sorted(set(d.model) & set(EXCLUDE))
    d = d[~d.model.isin(EXCLUDE)]
    d["family"] = d.model.map(FAMILY_OF)
    return d, excluded


def train_log(run_dir):
    p = ROOT / run_dir / "train_log.csv" if not Path(run_dir).is_absolute() else Path(run_dir) / "train_log.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    if "epoch" not in t:
        return None
    return t.groupby("epoch", as_index=False).last()  # detectron2 logs several rows per epoch


# ---------------------------------------------------------------------------------------------- charts
def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_ap_bars(det, path, title):
    det = det.sort_values("AP")
    n = len(det)
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * n + 0.9))
    h = 0.26
    y = range(n)
    for i, (col, lab) in enumerate([("AP50", "AP50"), ("AP75", "AP75"), ("AP", "AP50:95")]):
        ax.barh([v + (i - 1) * (h + 0.02) for v in y], det[col], height=h, color=BAR3[i], label=lab, zorder=3)
    for v, ap in zip(y, det.AP):
        ax.text(ap + 0.008, v + h + 0.02, f"{ap:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(list(y), [dn(m) for m in det.model])
    ax.set_xlim(0, 1)
    ax.set_xlabel("hodnota metriky (test)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", ncol=3)
    return _save(fig, path)


def chart_acc_latency(det, path, title):
    det = det[det.latency_e2e_ms_median.notna()]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for fam in FAMILY_COLOR:
        s = det[det.family == fam]
        if len(s):
            ax.scatter(s.latency_e2e_ms_median, s.AP, s=38, color=FAMILY_COLOR[fam], marker=FAMILY_MARKER[fam],
                       edgecolor="white", linewidth=1.5, label=fam, zorder=3)
    for _, r in det.iterrows():
        ax.annotate(dn(r.model), (r.latency_e2e_ms_median, r.AP), xytext=(5, 3), textcoords="offset points",
                    fontsize=6.3, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("latencia end-to-end, medián, batch 1 [ms] (log. mierka; orientačná)")
    ax.set_ylabel("AP50:95 (test)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    return _save(fig, path)


def chart_curves(runs, path, metric_label, families, kind="det"):
    """Small multiples: row 1 = normalised train loss per family, row 2 = validation metric per family."""
    fams = [f for f in families if any(FAMILY_OF.get(m) == f for m in runs)]
    if not fams:
        return None
    fig, axes = plt.subplots(2, len(fams), figsize=(7.4, 4.6), squeeze=False, sharey="row")
    for j, fam in enumerate(fams):
        ms = [m for m in FAMILIES[fam] if m in runs]
        for k, m in enumerate(ms):
            t = runs[m]
            c = SERIES[k % len(SERIES)]
            tl = t[["epoch", "train_loss"]].dropna()
            if len(tl):
                axes[0, j].plot(tl.epoch + 1, tl.train_loss / tl.train_loss.iloc[0], color=c, label=dn(m))
            vm = t[["epoch", "val_metric"]].dropna()
            if len(vm):
                axes[1, j].plot(vm.epoch + 1, vm.val_metric, color=c, label=dn(m))
        axes[0, j].set_title(fam, loc="left")
        axes[1, j].set_xlabel("epocha")
        axes[1, j].legend(loc="upper left", bbox_to_anchor=(0, -0.28), fontsize=6)
    axes[0, 0].set_ylabel("train loss / loss v 1. epoche")
    axes[1, 0].set_ylabel(metric_label)
    axes[1, 0].set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, path)


def chart_prec_rec(runs, path):
    """Validation precision and recall per epoch - logged only by Ultralytics (YOLO) and RF-DETR."""
    cols = {"yolo": ("metrics/precision(B)", "metrics/recall(B)"), "rfdetr": ("val/precision", "val/recall")}
    got = {}
    for m, t in runs.items():
        for key, (pc, rc) in cols.items():
            if m.startswith(key) and pc in t and rc in t:
                got[m] = t[["epoch", pc, rc]].dropna().rename(columns={pc: "P", rc: "R"})
    if not got:
        return None, []
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for k, (m, t) in enumerate(sorted(got.items(), key=lambda x: list(DISPLAY).index(x[0]))):
        c = SERIES[k % len(SERIES)]
        axes[0].plot(t.epoch + 1, t.P, color=c, label=dn(m))
        axes[1].plot(t.epoch + 1, t.R, color=c, label=dn(m))
    axes[0].set_title("validačná precision", loc="left")
    axes[1].set_title("validačný recall", loc="left")
    for a in axes:
        a.set_xlabel("epocha")
        a.set_ylim(0, 1)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=6, ncol=4)
    fig.tight_layout()
    return _save(fig, path), list(got)


# ---------------------------------------------------------------------------------------------- pdf bits
def styles():
    pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu",
                                  boldItalic="DejaVu-Bold")
    ss = getSampleStyleSheet()
    base = dict(fontName="DejaVu", textColor=colors.HexColor(INK))
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=18, leading=22,
                                alignment=TA_LEFT, spaceAfter=4),
        "sub": ParagraphStyle("s", fontName="DejaVu", fontSize=9.5, leading=13, textColor=colors.HexColor(INK2)),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=14, leading=17,
                             spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=11, leading=14,
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", **base, fontSize=9, leading=12.5, spaceAfter=5),
        "small": ParagraphStyle("sm", fontName="DejaVu", fontSize=7.5, leading=10, textColor=colors.HexColor(INK2)),
        "cell": ParagraphStyle("c", **base, fontSize=7.3, leading=9),
        "cellb": ParagraphStyle("cb", fontName="DejaVu-Bold", fontSize=7.3, leading=9),
        "bullet": ParagraphStyle("bl", **base, fontSize=9, leading=12.5, leftIndent=10, bulletIndent=0,
                                 spaceAfter=2),
    }


def table(rows, S, widths, header_rows=1, bold_col=None):
    data = [[Paragraph(str(c), S["cellb"] if i < header_rows else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    st = [("LINEBELOW", (0, 0), (-1, header_rows - 1), 0.8, colors.HexColor(INK2)),
          ("LINEBELOW", (0, header_rows), (-1, -1), 0.25, colors.HexColor(GRID)),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 3),
          ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if bold_col is not None:
        st.append(("BACKGROUND", (bold_col, header_rows), (bold_col, -1), colors.HexColor("#f3f2ee")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm=17.0):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def bullets(items, S):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


# ------------------------------------------------------------------------------------------ narrative
ARCH = [
    ("CNN (YOLO26, YOLOv10, Faster R-CNN)",
     ["Veľmi rýchla inferencia a malé modely (YOLO n/s: 2,5–10 M parametrov), vhodné na nasadenie na linke.",
      "Silná lokálna indukčná vložená vlastnosť konvolúcie – dobre sa učia na malých datasetoch.",
      "YOLO26 a YOLOv10 sú NMS-free (end-to-end) – jednoduchší a stabilnejší čas postprocessingu.",
      "Faster R-CNN je dobre preskúmaný, stabilný referenčný dvojstupňový detektor."],
     ["Obmedzený globálny kontext – ťažšie rozlišuje veľké, pretiahnuté alebo nízko-kontrastné defekty "
      "(napr. crazing, rolled-in scale, crease).",
      "Faster R-CNN: vysoké AP50, ale slabšia presnosť lokalizácie (AP75) a vyššia latencia pri vstupe 800–1333 px.",
      "Výsledok citlivo závisí od augmentácií a rozlíšenia vstupu (mozaika, letterbox)."]),
    ("CNN + attention (YOLOv12, Faster R-CNN + CBAM)",
     ["Attention (area attention v YOLOv12, kanálová a priestorová pozornosť v CBAM) zvýrazní informatívne kanály "
      "a oblasti bez veľkej zmeny architektúry.",
      "CBAM je lacný modul (+0,03 M parametrov) – umožňuje čisté meranie prínosu pozornosti pri inak rovnakej sieti."],
     ["Prínos je v našich dátach malý a pri jednom seede v rámci šumu; YOLOv12 nebol konzistentne lepší než YOLO26 "
      "rovnakej veľkosti (lepší len v 1 zo 4 porovnaní – NEU-DET, variant s).",
      "Attention bloky zvyšujú pamäťové nároky a latenciu (najmä pri vyššom rozlíšení).",
      "CBAM nie je detektor – hodnotí sa len ako doplnok k existujúcemu backbone/FPN."]),
    ("DETR rodina (RF-DETR, DEIMv2, RT-DETRv2, D-FINE)",
     ["Set-prediction bez NMS a bez kotiev; globálny kontext cez attention – dobré pri veľkých a nepravidelných defektoch.",
      "Najlepšia presnosť lokalizácie (AP75, AP50:95) na všetkých troch datasetoch.",
      "Silné predtrénované reprezentácie (DINOv2/DINOv3 v RF-DETR a DEIMv2) pomáhajú pri malých datasetoch.",
      "Real-time varianty (RT-DETRv2, RF-DETR-N) majú latenciu v rovnakom rádovom rozsahu ako YOLO modely "
      "(desiatky ms), na rozdiel od DINO + Swin-T."],
     ["Citlivejšie na hyperparametre a recept (lr, EMA, dĺžka tréningu, denoising) – ukázal to aj problém s D-FINE.",
      "Dlhší tréning (pomalšia konvergencia Hungarian matchingu) a vyššia pamäť ako YOLO rovnakej veľkosti.",
      "Pri veľmi malých objektoch môže byť obmedzujúci počet dotazov a rozlíšenie príznakov."]),
    ("ViT / Swin ako backbone (ViTDet, DINO + Swin-T, Cascade + Swin-T)",
     ["Hierarchické (Swin) alebo jednoduché (ViT) transformerové príznaky s veľkým receptívnym poľom.",
      "DINO + Swin-T: vysoké AR100 a konkurencieschopné AP na NEU-DET – silný detektor orientovaný na presnosť.",
      "Backbone je vymeniteľný – dá sa kombinovať s rôznymi detekčnými hlavami."],
     ["Najvyššia latencia a pamäť v štúdii (DINO + Swin-T ~ 5–20× pomalší ako YOLO); ViTDet-B potrebuje GPU s veľkou pamäťou.",
      "Na malých datasetoch potrebuje silné predtrénovanie; bez neho sa preučí.",
      "Zložitejšia inštalácia (kompilované CUDA operácie, Docker)."]),
    ("U-Net a hybridné segmentačné siete (U-Net, Attention U-Net, TransUNet, Swin-Unet)",
     ["Pixelová maska – presný tvar a plocha defektu (trhliny, póry), priamo merateľné AUPRO a pixel AUROC/AUPRC.",
      "Z mapy sa dá odvodiť skóre obrázka (anomália áno/nie) aj boxy (segmentation-derived detection).",
      "TransUNet/Swin-Unet pridávajú globálny kontext; Attention U-Net filtruje skip spojenia pozornostnými bránami."],
     ["Vyžadujú pixelové anotácie – v štúdii ich má iba Magnetic Tile.",
      "Boxy odvodené zo segmentácie majú nízke AP (fragmentácia a spájanie oblastí) – nie sú náhradou detektora.",
      "Transformerové hybridy sú ťažšie a pri malom datasete nie vždy lepšie než jednoduchý U-Net s predtrénovaným enkodérom."]),
]


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "reports" / "defectbench_report.pdf"))
    a = ap.parse_args()
    d, excluded = load_results(a.seed)
    build = ROOT / "reports" / "pdf_build"
    build.mkdir(parents=True, exist_ok=True)
    S = styles()
    story = []
    now = datetime.now()

    # ---------- title + key findings
    story += [Paragraph("Porovnanie architektúr na detekciu povrchových defektov", S["title"]),
              Paragraph(f"defectbench · priebežný report · {now:%d. %m. %Y %H:%M} · seed {a.seed} · "
                        f"{len(d)} dokončených a vyhodnotených behov", S["sub"]), Spacer(1, 8)]
    story.append(Paragraph(
        "<b>Dôležité:</b> ide o priebežné výsledky jedného tréningového seedu (42) – bez smerodajnej odchýlky. "
        "Rozdiely menšie než ~0,01–0,02 AP preto nemožno považovať za preukázané. Finálne čísla budú priemer ± std "
        "cez seedy 17, 42 a 73. Latencie boli merané počas paralelných tréningov na tej istej GPU (RTX A6000) "
        "a sú len orientačné.", S["body"]))

    det = d[d.task == "bbox"]
    seg = d[d.task == "seg"]
    best = {ds: det[det.dataset == ds].sort_values("AP", ascending=False) for ds in DATASETS}
    story.append(Paragraph("Hlavné zistenia", S["h2"]))
    kf = []
    tops = [f"{DATASETS[ds]}: {dn(b.iloc[0].model)} (AP50:95 {b.iloc[0].AP:.3f})" for ds, b in best.items() if len(b)]
    kf.append("Najvyššie AP50:95 dosahujú na všetkých datasetoch modely z <b>DETR rodiny</b> – " + "; ".join(tops) + ".")
    fam_ap75 = det.groupby(["dataset", "family"]).AP75.max().unstack()
    if {"CNN", "DETR rodina"} <= set(fam_ap75.columns):
        diffs = [f"{DATASETS[ds]} +{fam_ap75.loc[ds, 'DETR rodina'] - fam_ap75.loc[ds, 'CNN']:.3f}"
                 for ds in DATASETS if ds in fam_ap75.index and not fam_ap75.loc[ds, ["CNN", "DETR rodina"]].isna().any()]
        kf.append("Náskok transformerových detektorov je najmä v <b>presnosti lokalizácie (AP75)</b>: najlepší DETR vs. "
                  "najlepší CNN – " + ", ".join(diffs) + ". Pri AP50 sú rozdiely menšie.")
    fr = det[det.model.isin(["faster_rcnn_r50_fpn", "faster_rcnn_r50_fpn_cbam"])].pivot(index="dataset", columns="model", values="AP")
    if fr.shape[1] == 2:
        dd = [f"{DATASETS[ds]} {r.faster_rcnn_r50_fpn_cbam - r.faster_rcnn_r50_fpn:+.3f}" for ds, r in fr.dropna().iterrows()]
        kf.append("<b>CBAM</b> v Faster R-CNN mení AP50:95 len minimálne (" + ", ".join(dd) +
                  ") – pri jednom seede je to v rámci šumu.")
    yolo = det[det.model.str.startswith("yolo")]
    if len(yolo):
        kf.append(f"<b>YOLO modely</b> sú najrýchlejšie a najmenšie (2,5–10 M parametrov) a dosahujú AP50 blízko "
                  f"transformerov, ale s nižším AP75. Rozdiely medzi YOLO26/v10/v12 sú malé; menšie „n“ varianty často "
                  f"nie sú horšie než „s“ (malé datasety).")
    if len(seg):
        bs = seg.sort_values("AUPRO30", ascending=False).iloc[0]
        bd = seg.sort_values("Dice", ascending=False).iloc[0]
        kf.append(f"Na Magnetic Tile má najlepšie AUPRO <b>{dn(bs.model)}</b> ({bs.AUPRO30:.3f}) a najlepší Dice "
                  f"<b>{dn(bd.model)}</b> ({bd.Dice:.3f}). Pri rozpoznaní defektného obrázka (image-level AUROC) sú "
                  f"však detektory lepšie než segmentátory (najlepší detektor "
                  f"{det[det.dataset == 'magnetic_tile'].imgAUROC.max():.3f} vs. najlepší segmentátor {seg.imgAUROC.max():.3f}).")
    story += bullets(kf, S)
    if excluded:
        story.append(Paragraph("<b>Vynechané z porovnania:</b> " + " ".join(EXCLUDE[m] for m in excluded), S["small"]))

    # ---------- methodology
    story.append(Paragraph("Metodika v skratke", S["h2"]))
    story += bullets([
        "<b>Datasety:</b> NEU-DET (1 800 obrázkov, 6 tried, boxy), Magnetic Tile (1 340 obrázkov, 5 typov defektov + "
        "obrázky bez defektu, pixelové masky), GC10-DET (2 291 obrázkov, 10 tried, boxy). Skupinový stratifikovaný split "
        "cca 70/15/15 % (train/val/test) bez duplicít medzi časťami; všetky modely používajú rovnaký split.",
        "<b>Tréning:</b> oficiálne recepty a predtrénované váhy každého modelu, early stopping na validačnom AP, "
        "výber najlepšieho checkpointu na validácii; test sa použije iba raz.",
        "<b>AP50 / AP75 / AP50:95:</b> COCO average precision pri IoU 0,5 / 0,75 / priemer 0,5–0,95 (pycocotools, "
        "max. 100 detekcií na obrázok).",
        "<b>Recall@wp, Precision@wp:</b> v pracovnom bode – prah skóre zvolený na validácii (max F1, IoU ≥ 0,5) a "
        "zmrazený pre test. <b>AR100</b> = COCO priemerný recall (bez prahu skóre).",
        "<b>Image-level AUROC/AUPRC:</b> len Magnetic Tile (má obrázky bez defektu); skóre obrázka = max. skóre detekcie, "
        "pri segmentátore priemer 100 najvyšších pixelov. <b>Pixel AUROC/AUPRC a AUPRO</b> (FPR ≤ 0,3, normalizované) "
        "len pre segmentátory – z boxov sa pixelové mapy nevytvárajú.",
    ], S)

    # ---------- per dataset
    for ds, name in DATASETS.items():
        x = d[d.dataset == ds]
        xdet = x[x.task == "bbox"].sort_values("AP", ascending=False)
        xseg = x[x.task == "seg"].sort_values("AUPRO30", ascending=False)
        if not len(x):
            continue
        story.append(PageBreak())
        story.append(Paragraph(name, S["h1"]))
        if len(xdet):
            imgcols = ds == "magnetic_tile"
            hdr = ["model", "rodina", "AP50", "AP75", "AP50:<br/>95", "Prec@wp", "Recall@wp", "AR100"] + \
                  (["img AUROC", "img AUPRC"] if imgcols else []) + ["param. [M]", "latencia [ms]"]
            rows = [hdr]
            for _, r in xdet.iterrows():
                p = r.params_total
                rows.append([dn(r.model), r.family, fmt(r.AP50), fmt(r.AP75), f"<b>{fmt(r.AP)}</b>", fmt(r["P@wp"]),
                             fmt(r["R@wp"]), fmt(r.AR100)] + ([fmt(r.imgAUROC), fmt(r.imgAUPRC)] if imgcols else []) +
                            [fmt(float(p) / 1e6, 1) if str(p) not in ("nan", "N/A") else "–",
                             fmt(r.latency_e2e_ms_median, 1)])
            w = [3.9, 2.4] + [1.15] * 6 + ([1.25] * 2 if imgcols else []) + [1.2, 1.3]
            scale = 18.2 / sum(w)
            story.append(Paragraph(f"Detekcia – testovacia množina ({len(xdet)} modelov, zoradené podľa AP50:95)", S["h2"]))
            story.append(table(rows, S, [v * scale * cm for v in w], bold_col=4))
            story.append(Spacer(1, 6))
            story.append(img(chart_ap_bars(xdet, build / f"{ds}_ap.png", f"{name} – AP50, AP75 a AP50:95 (test)"), 16.5))
            story.append(img(chart_acc_latency(xdet, build / f"{ds}_lat.png",
                                               f"{name} – presnosť vs. latencia"), 16.5))
        if len(xseg):
            story.append(Paragraph(f"Segmentácia – testovacia množina ({len(xseg)} modelov, zoradené podľa AUPRO)", S["h2"]))
            rows = [["model", "Dice", "IoU", "pixel recall", "px AUROC", "px AUPRC", "AUPRO@0,3", "img AUROC",
                     "img AUPRC", "AP50 (seg→det)", "param. [M]"]]
            for _, r in xseg.iterrows():
                rows.append([dn(r.model), fmt(r.Dice), fmt(r.IoU), fmt(r.Rec), fmt(r.pxAUROC), fmt(r.pxAUPRC),
                             f"<b>{fmt(r.AUPRO30)}</b>", fmt(r.imgAUROC), fmt(r.imgAUPRC), fmt(r.s2d_AP50),
                             fmt(float(r.params_total) / 1e6, 1)])
            w = [4.0] + [1.3] * 10
            scale = 18.2 / sum(w)
            story.append(table(rows, S, [v * scale * cm for v in w], bold_col=6))

        # training curves
        det_runs = {r.model: t for _, r in xdet.iterrows() if (t := train_log(r.run_dir)) is not None}
        seg_runs = {r.model: t for _, r in xseg.iterrows() if (t := train_log(r.run_dir)) is not None}
        story.append(PageBreak())
        story.append(Paragraph(f"{name} – priebeh tréningu", S["h2"]))
        story.append(Paragraph(
            "Horný rad: tréningová loss vydelená hodnotou v 1. epoche (frameworky používajú rôzne definície loss, "
            "preto sa porovnáva iba relatívny trend). Dolný rad: validačná metrika po epochách tak, ako ju počíta "
            "framework počas tréningu (mAP50:95 na validačnej množine). Krivky končia v epoche early stoppingu.",
            S["small"]))
        p = chart_curves(det_runs, build / f"{ds}_curves_det.png", "val mAP50:95",
                         ["CNN", "CNN + attention", "DETR rodina", "ViT / Swin backbone"])
        if p:
            story.append(img(p, 17.5))
        p, pr_models = chart_prec_rec(det_runs, build / f"{ds}_pr.png")
        if p:
            story.append(Paragraph("Validačná precision a recall po epochách (logujú ich iba Ultralytics YOLO a RF-DETR; "
                                   "ostatné frameworky počas tréningu počítajú len mAP)", S["small"]))
            story.append(img(p, 17.5))
        if seg_runs:
            p = chart_curves(seg_runs, build / f"{ds}_curves_seg.png", "val Dice", ["U-Net a hybridy"])
            story.append(KeepTogether([Paragraph("Segmentátory – relatívna tréningová loss a validačný Dice", S["small"]),
                                       img(p, 12)]))
        # per-run qualitative figure for the best detector / segmenter
        if len(xdet):
            f = ROOT / xdet.iloc[0].run_dir / "figures" / "vis_test_errors.png"
            if f.exists():
                story.append(KeepTogether([Paragraph(
                    f"Ukážka predikcií najlepšieho detektora ({dn(xdet.iloc[0].model)}) na testovacej množine – "
                    "zelená GT, modrá TP, oranžová FP (typ chyby), červená FN.", S["small"]), img(f, 15.5)]))
        if len(xseg):
            f = ROOT / xseg.iloc[0].run_dir / "figures" / "vis_test_segmentation.png"
            if f.exists():
                story.append(KeepTogether([Paragraph(
                    f"Ukážka segmentácie ({dn(xseg.iloc[0].model)}) – vľavo predikcia (červená) s obrysom GT (zelená), "
                    "vpravo mapa pravdepodobnosti defektu.", S["small"]), img(f, 15.5)]))

    # ---------- architectures
    story.append(PageBreak())
    story.append(Paragraph("Výhody a nevýhody architektúr", S["h1"]))
    story.append(Paragraph(
        "Zhrnutie vychádza z vlastností architektúr a z nameraných výsledkov tohto (priebežného) behu. Pri tvrdeniach "
        "o rýchlosti platí, že latencie sú zatiaľ merané na zaťaženej GPU.", S["body"]))
    for title, pros, cons in ARCH:
        pro_p = [Paragraph("• " + t, S["cell"]) for t in pros]
        con_p = [Paragraph("• " + t, S["cell"]) for t in cons]
        t = Table([[Paragraph("<b>Výhody</b>", S["cellb"]), Paragraph("<b>Nevýhody</b>", S["cellb"])], [pro_p, con_p]],
                  colWidths=[9 * cm, 9 * cm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
        story.append(KeepTogether([Paragraph(title, S["h2"]), t, Spacer(1, 4)]))

    # family summary table (best AP50:95 per family and dataset)
    story.append(Paragraph("Najlepší model rodiny podľa AP50:95 (test)", S["h2"]))
    rows = [["rodina"] + list(DATASETS.values())]
    for fam in ["CNN", "CNN + attention", "DETR rodina", "ViT / Swin backbone"]:
        r = [fam]
        for ds in DATASETS:
            s = det[(det.dataset == ds) & (det.family == fam)].sort_values("AP", ascending=False)
            r.append(f"{dn(s.iloc[0].model)} – {s.iloc[0].AP:.3f}" if len(s) else "– (zatiaľ nedobehlo)")
        rows.append(r)
    story.append(table(rows, S, [3.6 * cm, 4.9 * cm, 4.9 * cm, 4.9 * cm]))

    # ---------- limitations / missing
    story.append(Paragraph("Obmedzenia a čo ešte chýba", S["h2"]))
    have = {(r.dataset, r.model) for _, r in d.iterrows()}
    missing = {}
    from defectbench.models.registry import load_registry, task_entry
    from defectbench.config import load_dataset_config
    reg = load_registry()
    for ds in DATASETS:
        cfg = load_dataset_config(ds)
        for m, mc in reg.items():
            for task in mc["tasks"]:
                if task_entry(mc, cfg, task)["status"] == "run" and (ds, m) not in have:
                    missing.setdefault(ds, []).append(dn(m))
    items = ["Jeden seed – žiadna smerodajná odchýlka; poradie modelov s rozdielom pod ~0,02 AP je neisté.",
             "Latencie z paralelného behu – finálna tabuľka latencií sa premeria na nezaťaženej GPU.",
             "Tréningové loss rôznych frameworkov nie sú číselne porovnateľné (rôzne zložky a váhy) – porovnáva sa trend."]
    for ds, ms in missing.items():
        items.append(f"<b>{DATASETS[ds]}</b> – zatiaľ bez výsledku: " + ", ".join(sorted(set(ms))) + ".")
    story += bullets(items, S)

    def on_page(c, doc):
        c.saveState()
        c.setFont("DejaVu", 7)
        c.setFillColor(colors.HexColor(INK2))
        c.drawString(1.5 * cm, 1.0 * cm, "defectbench – priebežný report (seed %d)" % a.seed)
        c.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, str(doc.page))
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="defectbench – porovnanie architektúr",
                            author="defectbench")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(a.out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
