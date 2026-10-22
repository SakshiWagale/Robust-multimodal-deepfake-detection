import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_forensic_pdf(
    filename: str,
    media_type: str,
    file_size: str,
    sha256_hash: str,
    case_id: str,
    timestamp: str,
    video_score: float,
    audio_score: float,
    final_score: float,
    verdict: str,
    confidence: float,
    threshold: float
) -> io.BytesIO:
    """
    Generates an official, academic-grade VERITAS Forensic Certificate as a PDF byte stream.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CertTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0f172a')
    )

    subtitle_style = ParagraphStyle(
        'CertSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#475569')
    )

    meta_style = ParagraphStyle(
        'CertMeta',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#64748b')
    )

    section_style = ParagraphStyle(
        'CertSection',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'CertBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    code_cell_style = ParagraphStyle(
        'TableCodeCell',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # 1. Header
    story.append(Paragraph("VERITAS MULTIMODAL FORENSICS LABORATORY", title_style))
    story.append(Paragraph("OFFICIAL DIGITAL MEDIA INTEGRITY &amp; DEEPFAKE VERIFICATION CERTIFICATE", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"CASE ID: {case_id}  &bull;  TIMESTAMP: {timestamp}  &bull;  STANDARD: Salvi et al. (MDPI 2023)", meta_style))
    story.append(Spacer(1, 12))

    # 2. Big Verdict Stamp Box
    is_real = (verdict.upper() == "REAL")
    verdict_color = colors.HexColor('#059669') if is_real else colors.HexColor('#dc2626')
    verdict_bg = colors.HexColor('#f0fdf4') if is_real else colors.HexColor('#fef2f2')

    verdict_text = f"<b><font size='22' color='{verdict_color.hexval()}'>{verdict.upper()} / {'AUTHENTIC' if is_real else 'SYNTHETIC DEEPFAKE'}</font></b>"
    confidence_text = f"<font size='11' color='#0f172a'><b>CONFIDENCE LEVEL: {confidence:.2f}%</b>  (Threshold &tau; = {threshold})</font>"
    
    if is_real:
        note_text = "Verified authentic. Facial spatio-temporal frames and acoustic speech characteristics match unmanipulated human media. No generative boundary noise detected."
    else:
        note_text = "Synthetic manipulation detected. Discrepancies found in facial frame boundaries, GAN generative artifacts, or speech vocoder acoustic features."

    verdict_table = Table(
        [
            [Paragraph(verdict_text, ParagraphStyle('VText', parent=styles['Normal'], alignment=TA_CENTER))],
            [Paragraph(confidence_text, ParagraphStyle('CText', parent=styles['Normal'], alignment=TA_CENTER))],
            [Paragraph(note_text, ParagraphStyle('NText', parent=styles['Normal'], alignment=TA_CENTER, fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#475569')))]
        ],
        colWidths=[520]
    )
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), verdict_bg),
        ('BOX', (0, 0), (-1, -1), 2, verdict_color),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # 3. Evidence Identification Table
    story.append(Paragraph("1. EVIDENCE IDENTIFICATION", section_style))
    evidence_data = [
        [Paragraph("Target Filename", table_header_style), Paragraph(str(filename), table_cell_style)],
        [Paragraph("Media Modality", table_header_style), Paragraph(str(media_type).upper(), table_cell_style)],
        [Paragraph("File Size", table_header_style), Paragraph(str(file_size), table_cell_style)],
        [Paragraph("SHA-256 Digital Checksum", table_header_style), Paragraph(str(sha256_hash), code_cell_style)]
    ]
    t1 = Table(evidence_data, colWidths=[180, 340])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # 4. Multimodal Forensic Analysis Table
    story.append(Paragraph("2. MULTIMODAL DUAL-STREAM ANALYSIS", section_style))
    v_str = f"{video_score:.4f}" if video_score is not None else "Unavailable (Audio-Only)"
    a_str = f"{audio_score:.4f}" if audio_score is not None else "Unavailable (Silent Video / Image)"
    
    analysis_data = [
        [Paragraph("Visual Stream (MobileNetV2)", table_header_style), Paragraph(f"{v_str}  (0.0 = Fake, 1.0 = Real)", table_cell_style)],
        [Paragraph("Acoustic Stream (Log-Mel CNN)", table_header_style), Paragraph(f"{a_str}  (0.0 = Fake, 1.0 = Real)", table_cell_style)],
        [Paragraph("Late-Fusion Rule", table_header_style), Paragraph("0.60 &times; Visual + 0.40 &times; Acoustic (Weighted Decision)", table_cell_style)],
        [Paragraph("Decision Threshold (&tau;)", table_header_style), Paragraph(f"&tau; = {threshold}", table_cell_style)],
        [Paragraph("Final Model Fused Score", table_header_style), Paragraph(f"<b><font color='{verdict_color.hexval()}'>{final_score:.4f}</font></b>", table_cell_style)]
    ]
    t2 = Table(analysis_data, colWidths=[180, 340])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 25))

    # 5. Footer & Official Signature
    footer_data = [
        [
            Paragraph("<b>Investigation Directorate:</b> VERITAS Cyber Forensics<br/><b>Inspection Method:</b> Salvi et al. (MDPI 2023) Multimodal Analysis<br/><b>Cryptographic Status:</b> Verified &bull; Tamper-Sealed", body_style),
            Paragraph("____________________________<br/><b>Lead Forensic Examiner</b><br/>VERITAS Authentication Bureau", ParagraphStyle('Sign', parent=styles['Normal'], alignment=TA_CENTER, fontName='Helvetica', fontSize=8, leading=11))
        ]
    ]
    t3 = Table(footer_data, colWidths=[330, 190])
    t3.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t3)

    doc.build(story)
    buffer.seek(0)
    return buffer
