import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_patient_pdf(patient, predictions, filepath):
    """
    Generates a beautifully styled, clinical-grade medical assessment report.
    - patient: SQLAlchemy PatientRecord model or dict
    - predictions: List of dicts representing prediction outcomes
    - filepath: output destination
    """
    # Page setup
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom color palette
    primary_color = colors.HexColor("#0D5C75")   # Dark Teal
    secondary_color = colors.HexColor("#1982A1") # Medium Blue
    accent_color = colors.HexColor("#D1ECF1")    # Light Alert Blue
    text_color = colors.HexColor("#212529")      # Dark Grey
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        spaceAfter=4
    )
    
    bold_style = ParagraphStyle(
        'BoldText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=text_color,
        spaceAfter=4
    )
    
    alert_style = ParagraphStyle(
        'AlertText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.HexColor("#0f5132")
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("AI-Powered Clinical Decision Support System", title_style))
    story.append(Paragraph("<b>PATIENT RISK ASSESSMENT REPORT</b>", ParagraphStyle('Sub', parent=normal_style, fontSize=12, textColor=secondary_color)))
    story.append(Spacer(1, 10))
    
    # Decorative line
    line_table = Table([[""]], colWidths=[530], rowHeights=[2])
    line_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 15))
    
    # Section 1: Patient Demographics
    story.append(Paragraph("Patient Demographics", section_style))
    
    # Determine attributes
    gender_str = "Male" if int(getattr(patient, 'gender', 1)) == 1 else "Female"
    weight = f"{getattr(patient, 'weight_kg', 'N/A')} kg"
    height = f"{getattr(patient, 'height_cm', 'N/A')} cm"
    bmi = f"{getattr(patient, 'bmi', 'N/A')}"
    
    demo_data = [
        [Paragraph("<b>Patient Name:</b>", normal_style), Paragraph(str(getattr(patient, 'name', 'Unknown')), normal_style),
         Paragraph("<b>Date of Report:</b>", normal_style), Paragraph(datetime_datetime_now_str(), normal_style)],
        [Paragraph("<b>Age:</b>", normal_style), Paragraph(f"{int(getattr(patient, 'age', 0))} years", normal_style),
         Paragraph("<b>Gender:</b>", normal_style), Paragraph(gender_str, normal_style)],
        [Paragraph("<b>Weight:</b>", normal_style), Paragraph(weight, normal_style),
         Paragraph("<b>Height:</b>", normal_style), Paragraph(height, normal_style)],
        [Paragraph("<b>Calculated BMI:</b>", normal_style), Paragraph(bmi, bold_style),
         Paragraph("<b>Smoking Status:</b>", normal_style), Paragraph("Yes" if getattr(patient, 'smoking_status', 0) else "No", normal_style)]
    ]
    
    demo_table = Table(demo_data, colWidths=[100, 160, 100, 170])
    demo_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.lightgrey),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 15))
    
    # Section 2: Clinical Vitals & Lab Measurements
    story.append(Paragraph("Vitals & Laboratory Profile", section_style))
    
    systolic = getattr(patient, 'systolic_bp', 120)
    diastolic = getattr(patient, 'diastolic_bp', 80)
    bp = f"{systolic}/{diastolic} mmHg"
    glucose = f"{getattr(patient, 'glucose', 'N/A')} mg/dL"
    chol = f"{getattr(patient, 'cholesterol', 'N/A')} mg/dL"
    hr = f"{getattr(patient, 'heart_rate', 'N/A')} bpm"
    ins = f"{getattr(patient, 'insulin', 'N/A')} μU/mL"
    cre = f"{getattr(patient, 'creatinine', 'N/A')} mg/dL"
    hba = f"{getattr(patient, 'hba1c', 'N/A')} %"
    
    measure_data = [
        [Paragraph("<b>Blood Pressure:</b>", normal_style), Paragraph(bp, normal_style),
         Paragraph("<b>Fasting Glucose:</b>", normal_style), Paragraph(glucose, normal_style)],
        [Paragraph("<b>Heart Rate:</b>", normal_style), Paragraph(hr, normal_style),
         Paragraph("<b>HbA1c Level:</b>", normal_style), Paragraph(hba, normal_style)],
        [Paragraph("<b>Total Cholesterol:</b>", normal_style), Paragraph(chol, normal_style),
         Paragraph("<b>Insulin Level:</b>", normal_style), Paragraph(ins, normal_style)],
        [Paragraph("<b>Serum Creatinine:</b>", normal_style), Paragraph(cre, normal_style),
         Paragraph("<b>Oxygen Saturation:</b>", normal_style), Paragraph(f"{getattr(patient, 'oxygen_saturation', 'N/A')}%", normal_style)]
    ]
    
    measure_table = Table(measure_data, colWidths=[110, 150, 110, 160])
    measure_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.lightgrey),
    ]))
    story.append(measure_table)
    story.append(Spacer(1, 15))
    
    # Section 3: Diagnostic Risk Assessment
    story.append(Paragraph("AI Disease Risk Diagnostics Summary", section_style))
    
    risk_headers = [
        Paragraph("<b>Disease Condition</b>", bold_style), 
        Paragraph("<b>Probability</b>", bold_style), 
        Paragraph("<b>Risk Category</b>", bold_style), 
        Paragraph("<b>Confidence Level</b>", bold_style)
    ]
    
    risk_rows = [risk_headers]
    for pred in predictions:
        prob = pred.get('probability', 0.0)
        risk_lvl = pred.get('risk_level', 'Low')
        
        # Color coding risk level
        if risk_lvl == "High":
            cat_text = f"<font color='red'><b>{risk_lvl}</b></font>"
        elif risk_lvl == "Moderate":
            cat_text = f"<font color='orange'><b>{risk_lvl}</b></font>"
        else:
            cat_text = f"<font color='green'><b>{risk_lvl}</b></font>"
            
        risk_rows.append([
            Paragraph(pred.get('disease_name', 'Unknown'), normal_style),
            Paragraph(f"{prob * 100:.1f}%", normal_style),
            Paragraph(cat_text, normal_style),
            Paragraph(f"{pred.get('confidence', 0.90) * 100:.0f}%", normal_style)
        ])
        
    risk_table = Table(risk_rows, colWidths=[180, 100, 120, 130])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    # Quick fix for text color in header row
    for col_idx in range(len(risk_headers)):
        risk_table.setStyle(TableStyle([
            ('TEXTCOLOR', (col_idx, 0), (col_idx, 0), colors.white),
        ]))
    story.append(risk_table)
    story.append(Spacer(1, 15))
    
    # Section 4: Clinical Guidance & Contributing Factors
    story.append(Paragraph("Explainable AI (XAI) Risk Factors & Guidance", section_style))
    
    guidance_blocks = []
    has_high_risk = False
    
    for pred in predictions:
        risk_lvl = pred.get('risk_level', 'Low')
        if risk_lvl in ['High', 'Moderate']:
            has_high_risk = True
            disease_name = pred.get('disease_name', 'Condition')
            factors = pred.get('contributing_factors', [])
            recs = pred.get('recommendation', 'Consult healthcare provider.')
            
            guidance_text = f"<b>{disease_name} Risk factors:</b> "
            if factors:
                factor_strs = [f"{f.get('feature', 'marker')} ({f.get('impact', 'elevated')})" for f in factors[:3]]
                guidance_text += ", ".join(factor_strs)
            else:
                guidance_text += "Elevated clinical markers."
                
            guidance_blocks.append([
                Paragraph(guidance_text, normal_style),
                Paragraph(f"<b>Recommended Action:</b> {recs}", alert_style)
            ])
            
    if not has_high_risk:
        guidance_blocks.append([
            Paragraph("No immediate elevated risk markers detected.", normal_style),
            Paragraph("<b>Recommended Action:</b> Maintain a balanced lifestyle, exercise regularly, and schedule regular annual health physicals.", alert_style)
        ])
        
    guidance_table = Table(guidance_blocks, colWidths=[240, 290])
    guidance_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E3E5")),
    ]))
    story.append(guidance_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.gray,
        spaceBefore=15
    )
    story.append(Paragraph(
        "<b>Disclaimer:</b> This clinical report is powered by machine learning algorithms and explainable AI modules. "
        "It is designed to assist medical professionals as a decision support aid and does not constitute formal diagnostic advice. "
        "All predictions should be verified clinically by licensed healthcare providers.", 
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)

def datetime_datetime_now_str():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
