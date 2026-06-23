import os
import re

GUIDELINES_PATH = "explainability/guidelines.txt"

class RAGService:
    _guidelines_cache = {}
    
    @classmethod
    def load_guidelines(cls):
        """Loads and parses the guidelines database into structured memory sections."""
        if cls._guidelines_cache:
            return cls._guidelines_cache
            
        if not os.path.exists(GUIDELINES_PATH):
            return {}
            
        with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
        sections = re.split(r'\[(.*?)\]', content)
        # The split returns: ['', 'CARDIAC_DISEASE_GUIDELINES', 'Condition: ...', 'DIABETES_GUIDELINES', ...]
        
        parsed = {}
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i+1].strip() if (i+1) < len(sections) else ""
            
            # Parse body lines into dict
            body_dict = {}
            for line in body.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    body_dict[k.strip().lower()] = v.strip()
            parsed[header] = body_dict
            
        cls._guidelines_cache = parsed
        return parsed

    @classmethod
    def search_guidelines(cls, query):
        """Matches user query against guidelines sections using keyword scores."""
        guidelines = cls.load_guidelines()
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for name, info in guidelines.items():
            score = 0
            # Condition match
            condition = info.get('condition', '').lower()
            if condition in query_lower:
                score += 10
                
            # Symptoms matching
            symptoms = info.get('key symptoms', '').lower()
            for sym in symptoms.split(','):
                if sym.strip() in query_lower and len(sym.strip()) > 3:
                    score += 2
                    
            # Dietary or advice keywords matching
            diet = info.get('dietary advice', '').lower()
            for wd in ['diet', 'eat', 'food', 'nutrition', 'sodium', 'sugar', 'cholesterol', 'protein', 'potassium']:
                if wd in query_lower and wd in diet:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_match = (name, info)
                
        return best_match

    @classmethod
    def answer_query(cls, query, patient_profile=None):
        """
        Processes query, retrieves matching clinical guidance,
        and constructs an integrated clinical response.
        """
        match = cls.search_guidelines(query)
        
        # Check if query contains general symptom keywords for a quick checker
        symptom_checker_response = cls.run_symptom_checker(query)
        
        if not match:
            # Fallback when no direct disease guidelines match
            resp = "### Clinical Assistant Response\n\n"
            if symptom_checker_response:
                resp += symptom_checker_response + "\n\n"
            else:
                resp += "I can assist with clinical inquiries regarding Heart Disease, Diabetes, Breast Cancer, Chronic Kidney Disease, Liver Disease, and Hypertension.\n\n"
                
            resp += "Please specify a symptom or condition (e.g., 'What is the recommended diet for a patient with high blood pressure?' or 'Explain liver disease symptoms').\n"
            resp += "\n*Disclaimer: Decisions must be clinically verified by a licensed doctor.*"
            return resp
            
        section_name, info = match
        condition = info.get('condition', 'the matched condition')
        symptoms = info.get('key symptoms', 'N/A')
        markers = info.get('clinical markers', 'N/A')
        diet = info.get('dietary advice', 'N/A')
        preventive = info.get('preventive recommendations', 'N/A')
        action = info.get('clinical action', 'N/A')
        
        resp = f"### Clinical Guidelines: {condition}\n\n"
        
        if symptom_checker_response:
            resp += f"⚠️ **Symptom Alert:** {symptom_checker_response}\n\n"
            
        # Integrate patient stats if provided
        if patient_profile:
            resp += f"🧑‍⚕️ **Patient Profile Analysis ({patient_profile.get('name', 'Subject')}):**\n"
            if condition == "Diabetes Mellitus":
                gl = patient_profile.get('glucose')
                if gl:
                    resp += f"- Patient's Glucose level is **{gl} mg/dL** (Standard Threshold: Prediabetes >100, Diabetes >125).\n"
            elif condition == "Hypertension (High Blood Pressure)":
                sys = patient_profile.get('systolic_bp')
                dia = patient_profile.get('diastolic_bp')
                if sys and dia:
                    resp += f"- Patient's Blood Pressure is **{sys}/{dia} mmHg** (Threshold: Hypertension >=130/80).\n"
            elif condition == "Chronic Kidney Disease (CKD)":
                cre = patient_profile.get('creatinine')
                if cre:
                    resp += f"- Patient's Serum Creatinine level is **{cre} mg/dL** (Threshold: Normal ~0.6 - 1.2).\n"
            elif condition == "Liver Disease":
                tb = patient_profile.get('total_bilirubin')
                if tb:
                    resp += f"- Patient's Total Bilirubin is **{tb} mg/dL** (Threshold: Normal <1.2).\n"
            resp += "\n"
            
        resp += f"🎯 **Key Indicators & Thresholds:**\n{markers}\n\n"
        resp += f"🥗 **Dietary Guidelines:**\n{diet}\n\n"
        resp += f"🛡️ **Preventive Interventions:**\n{preventive}\n\n"
        resp += f"⚕️ **Required Clinical Action:**\n{action}\n\n"
        
        resp += "---\n*Clinical Decision Support Tool - Verify all recommendations independently before initiating treatment.*"
        return resp

    @staticmethod
    def run_symptom_checker(query):
        """Identifies symptoms in text query and returns a preliminary severity warning."""
        q = query.lower()
        warnings = []
        
        if "chest pain" in q or "angina" in q or "pressure in chest" in q:
            warnings.append("**Chest Pain**: This is a high-severity symptom. If accompanied by shortness of breath, dizziness, or pain radiating to the jaw/arm, refer the patient to emergency cardiac services immediately.")
        if "frequent urination" in q or "excessive thirst" in q:
            warnings.append("**Polyuria / Polydipsia**: Common indicators of hyperglycemia and diabetes mellitus. Recommend fasting plasma glucose or HbA1c screening.")
        if "shortness of breath" in q or "dyspnea" in q:
            warnings.append("**Shortness of Breath**: Indicates potential respiratory or cardiovascular compromise. Monitor oxygen saturation level (SpO2) and heart rate.")
        if "yellow skin" in q or "yellow eyes" in q or "jaundice" in q:
            warnings.append("**Jaundice**: Suggests elevated bilirubin and liver dysfunction. Prompt liver panel (LFT) and abdominal ultrasound are recommended.")
        if "foamy urine" in q or "puffy feet" in q:
            warnings.append("**Proteinuria / Edema indicators**: Warning signs of kidney damage and fluid retention. Recommend urinalysis for protein and serum creatinine GFR calculation.")
            
        if warnings:
            return "\n\n".join(warnings)
        return None
