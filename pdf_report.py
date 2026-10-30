from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def generate_health_report(
    disease,
    prediction,
    inputs,
    patient_name="Not provided",
    shap_explanation=None,
):
    """
    Generate a downloadable PDF report for a Health Guard prediction.

    Parameters:
        disease: Name of the disease prediction module.
        prediction: Prediction result as text.
        inputs: Dictionary containing the input feature names and values.
        patient_name: Optional patient name.
        shap_explanation: Optional list of SHAP explanation strings.

    Returns:
        BytesIO object containing the generated PDF.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "HealthGuardTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#166534"),
        fontSize=22,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "HealthGuardSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        fontSize=10,
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "HealthGuardSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#166534"),
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    story = []

    story.append(Paragraph("HEALTH GUARD", title_style))
    story.append(
        Paragraph(
            "Multiple Disease Analysis and Prediction System",
            subtitle_style,
        )
    )

    story.append(Paragraph("Health Report Summary", section_style))

    summary_data = [
        ["Report Date", datetime.now().strftime("%d-%m-%Y %I:%M %p")],
        ["Patient Name", str(patient_name)],
        ["Prediction Module", str(disease)],
        ["Prediction Result", str(prediction)],
    ]

    summary_table = Table(summary_data, colWidths=[45 * mm, 110 * mm])

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dcfce7")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#166534")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(summary_table)

    story.append(Paragraph("Input Values", section_style))

    input_data = [["Feature", "Entered Value"]]

    for feature, value in inputs.items():
        input_data.append([str(feature), str(value)])

    input_table = Table(
        input_data,
        colWidths=[75 * mm, 80 * mm],
        repeatRows=1,
    )

    input_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#166534")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(input_table)

    if shap_explanation:
        story.append(Paragraph("Explainable AI (SHAP)", section_style))

        for explanation in shap_explanation:
            story.append(
                Paragraph(
                    str(explanation).replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"),
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "<b>Important Disclaimer:</b> This report contains "
            "experimental machine-learning predictions for informational "
            "and educational purposes only. It is not a medical diagnosis "
            "and must not replace consultation with a qualified healthcare "
            "professional.",
            styles["BodyText"],
        )
    )

    document.build(story)

    buffer.seek(0)
    return buffer
