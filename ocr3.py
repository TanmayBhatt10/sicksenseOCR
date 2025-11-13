import cv2
import fitz
import numpy as np
import pytesseract
import os
import re
from pdf2image import convert_from_path
from pathlib import Path
import json
import fitz


# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class MedicalReportAnalyzer:
    def __init__(self, poppler_path=None):
        
        self.poppler_path = poppler_path
        
        # Comprehensive disease indicators with specific keywords for each disease
        self.disease_keywords = {
            # Cancer indicators
            'cancer': {
                'keywords': ['cancer', 'malignancy', 'malignant', 'tumor', 'neoplasm', 
                           'carcinoma', 'metastasis', 'metastatic', 'suspicious nodules',
                           'adenocarcinoma', 'sarcoma', 'lymphoma', 'leukemia'],
                'markers': ['CA 15.3', 'CA 125', 'CEA', 'PSA', 'AFP', 'CA 19-9'],
                'findings': ['mass', 'lesion', 'growth', 'abnormal cells']
            },
            
            # Diabetes indicators
            'diabetes': {
                'keywords': ['diabetes', 'diabetic', 'hyperglycemia', 'insulin resistance',
                           'type 1 diabetes', 'type 2 diabetes', 'prediabetes', 'diabetic ketoacidosis'],
                'markers': ['glucose', 'HbA1c', 'fasting blood sugar', 'random blood sugar',
                          'postprandial glucose', 'OGTT', 'insulin', 'C-peptide'],
                'thresholds': {
                    'glucose_fasting': 126,  # mg/dL
                    'glucose_random': 200,
                    'HbA1c': 6.5  # percentage
                },
                'findings': ['elevated glucose', 'high blood sugar', 'impaired glucose tolerance',
                           'glycosuria', 'ketonuria']
            },
            
            # Heart Disease indicators
            'heart_disease': {
                'keywords': ['cardiac', 'heart disease', 'coronary artery disease', 'CAD',
                           'myocardial infarction', 'heart attack', 'angina', 'heart failure',
                           'congestive heart failure', 'CHF', 'arrhythmia', 'atrial fibrillation',
                           'cardiomyopathy', 'valvular disease', 'ischemic heart disease'],
                'markers': ['troponin', 'CK-MB', 'BNP', 'NT-proBNP', 'LDH', 'myoglobin',
                          'cholesterol', 'LDL', 'HDL', 'triglycerides'],
                'thresholds': {
                    'troponin': 0.04,  # ng/mL
                    'total_cholesterol': 240,  # mg/dL
                    'LDL': 160,
                    'triglycerides': 200
                },
                'findings': ['ST elevation', 'Q waves', 'ejection fraction reduced',
                           'wall motion abnormality', 'valve dysfunction', 'cardiomegaly',
                           'coronary stenosis', 'blockage']
            },
            
            # Liver Disease indicators
            'liver_disease': {
                'keywords': ['hepatitis', 'cirrhosis', 'fatty liver', 'NAFLD', 'NASH',
                           'liver disease', 'hepatic', 'jaundice', 'ascites', 'hepatomegaly',
                           'liver failure', 'portal hypertension', 'hepatocellular carcinoma'],
                'markers': ['ALT', 'AST', 'ALP', 'GGT', 'bilirubin', 'albumin', 'PT', 'INR',
                          'SGOT', 'SGPT', 'direct bilirubin', 'indirect bilirubin'],
                'thresholds': {
                    'ALT': 56,  # U/L
                    'AST': 40,
                    'bilirubin_total': 1.2,  # mg/dL
                    'ALP': 147
                },
                'findings': ['elevated liver enzymes', 'hepatic steatosis', 'fibrosis',
                           'esophageal varices', 'splenomegaly']
            },
            
            # Chronic Kidney Disease indicators
            'kidney_disease': {
                'keywords': ['chronic kidney disease', 'CKD', 'renal failure', 'kidney failure',
                           'nephropathy', 'glomerulonephritis', 'nephrotic syndrome',
                           'renal insufficiency', 'end stage renal disease', 'ESRD',
                           'acute kidney injury', 'AKI', 'diabetic nephropathy'],
                'markers': ['creatinine', 'BUN', 'eGFR', 'urea', 'albumin', 'protein',
                          'uric acid', 'potassium', 'phosphorus', 'calcium'],
                'thresholds': {
                    'creatinine': 1.3,  # mg/dL
                    'BUN': 20,  # mg/dL
                    'eGFR': 60,  # mL/min (below this indicates CKD)
                    'albumin_urine': 30  # mg/g (microalbuminuria)
                },
                'findings': ['reduced GFR', 'proteinuria', 'hematuria', 'elevated creatinine',
                           'fluid retention', 'electrolyte imbalance', 'anemia']
            },
            
            # Breast Cancer indicators
            'breast_cancer': {
                'keywords': ['breast cancer', 'breast carcinoma', 'ductal carcinoma', 'DCIS',
                           'invasive ductal carcinoma', 'lobular carcinoma', 'breast mass',
                           'breast tumor', 'mammographic abnormality'],
                'markers': ['CA 15-3', 'CA 27.29', 'CEA', 'HER2', 'ER', 'PR',
                          'estrogen receptor', 'progesterone receptor'],
                'findings': ['breast mass', 'microcalcifications', 'architectural distortion',
                           'nipple discharge', 'skin changes', 'lymph node involvement',
                           'BIRADS 4', 'BIRADS 5', 'suspicious lesion']
            },
            
            # Hepatitis indicators
            'hepatitis': {
                'keywords': ['hepatitis A', 'hepatitis B', 'hepatitis C', 'hepatitis D',
                           'hepatitis E', 'viral hepatitis', 'acute hepatitis',
                           'chronic hepatitis', 'HBV', 'HCV', 'HAV'],
                'markers': ['HBsAg', 'anti-HBs', 'anti-HCV', 'HCV RNA', 'HBV DNA',
                          'anti-HAV IgM', 'anti-HAV IgG', 'HBeAg', 'anti-HBc'],
                'findings': ['hepatitis positive', 'viral load detectable', 'reactive',
                           'elevated transaminases', 'liver inflammation']
            },
            
            # General positive findings
            'positive_findings': {
                'keywords': ['positive', 'detected', 'abnormal', 'elevated', 'increased',
                           'outside reference range', 'significant', 'remarkable',
                           'pathological', 'acute', 'severe', 'critical', 'high risk']
            }
        }
        
        # Negative indicators
        self.negative_keywords = [
            'negative', 'normal', 'no evidence', 'no such symptoms', 'no trace',
            'within normal limits', 'unremarkable', 'no abnormality', 'healthy',
            'clear', 'no suspicious', 'non-reactive', 'absent', 'rule out',
            'no significant', 'stable', 'well controlled'
        ]
        
        # Clinical sections to prioritize
        self.clinical_sections = [
            'outcome', 'conclusions', 'findings', 'results', 'diagnosis',
            'impression', 'summary', 'interpretation', 'assessment', 'opinion'
        ]
    
    def preprocess_image(self, image):
        """Enhance image quality for better OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        
        processed = self.preprocess_image(img)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config, lang='eng')
        return text
    
    import fitz  # PyMuPDF

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF using PyMuPDF (no Poppler needed).
        Returns the full text as a single string.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)  # Open the PDF
            full_text = ""
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                full_text += f"\n--- Page {page_num} ---\n{text}"
            
            doc.close()
            return full_text
        
        except Exception as e:
            raise Exception(f"Could not read PDF with PyMuPDF. Error: {e}")

    
    def extract_clinical_values(self, text):
        """Extract clinical values and biomarkers"""
        results = {}
        text_upper = text.upper()
        
        # Common patterns for lab values
        patterns = {
            # Diabetes markers
            'glucose': r'(?:GLUCOSE|SUGAR|FBS|RBS|PPBS)[\s:]*(\d+\.?\d*)',
            'HbA1c': r'(?:HBA1C|HBALC|A1C|GLYCATED\s*HEMOGLOBIN)[\s:]*(\d+\.?\d*)',
            
            # Heart disease markers
            'troponin': r'(?:TROPONIN|TROP[ONIN]*\s*[IT])[\s:]*(\d+\.?\d*)',
            'cholesterol': r'(?:CHOLESTEROL|CHOL)[\s:]*(\d+\.?\d*)',
            'LDL': r'(?:LDL)[\s:]*(\d+\.?\d*)',
            'HDL': r'(?:HDL)[\s:]*(\d+\.?\d*)',
            'triglycerides': r'(?:TRIGLYCERIDES|TG)[\s:]*(\d+\.?\d*)',
            
            # Liver markers
            'ALT': r'(?:ALT|SGPT)[\s:]*(\d+\.?\d*)',
            'AST': r'(?:AST|SGOT)[\s:]*(\d+\.?\d*)',
            'bilirubin': r'(?:BILIRUBIN|BILI)[\s:]*(\d+\.?\d*)',
            'ALP': r'(?:ALP|ALKALINE\s*PHOSPHATASE)[\s:]*(\d+\.?\d*)',
            
            # Kidney markers
            'creatinine': r'(?:CREATININE|CREAT)[\s:]*(\d+\.?\d*)',
            'BUN': r'(?:BUN|UREA)[\s:]*(\d+\.?\d*)',
            'eGFR': r'(?:EGFR|GFR)[\s:]*(\d+\.?\d*)',
            
            # Cancer markers
            'CEA': r'(?:CEA)[\s:]*(\d+\.?\d*)',
            'CA_15_3': r'(?:CA\s*15[\.-]3)[\s:]*(\d+\.?\d*)',
            'CA_125': r'(?:CA\s*125)[\s:]*(\d+\.?\d*)',
            'PSA': r'(?:PSA)[\s:]*(\d+\.?\d*)',
        }
        
        for marker, pattern in patterns.items():
            matches = re.findall(pattern, text_upper)
            if matches:
                try:
                    results[marker] = float(matches[0])
                except ValueError:
                    pass
        
        return results
    
    def check_threshold_violations(self, clinical_values, disease_type):
        """Check if clinical values exceed disease-specific thresholds"""
        violations = []
        
        if disease_type not in self.disease_keywords:
            return violations
        
        disease_data = self.disease_keywords[disease_type]
        if 'thresholds' not in disease_data:
            return violations
        
        thresholds = disease_data['thresholds']
        
        for marker, threshold in thresholds.items():
            if marker in clinical_values:
                value = clinical_values[marker]
                
                # Special handling for eGFR (lower is worse for kidneys)
                if marker == 'eGFR':
                    if value < threshold:
                        violations.append(f"{marker}: {value} (below normal {threshold})")
                else:
                    if value > threshold:
                        violations.append(f"{marker}: {value} (above normal {threshold})")
        
        return violations
    
    def analyze_disease_indicators(self, text):
        """Comprehensive analysis for multiple diseases"""
        text_lower = text.lower()
        
        analysis = {
            'diseases_detected': [],
            'all_findings': {},
            'clinical_values': {},
            'threshold_violations': {},
            'overall_status': 'negative',
            'confidence': 'low',
            'recommendations': []
        }
        
        # Extract clinical values
        clinical_values = self.extract_clinical_values(text)
        analysis['clinical_values'] = clinical_values
        
        # Check each disease type
        for disease_type, disease_data in self.disease_keywords.items():
            if disease_type == 'positive_findings':
                continue
                
            disease_analysis = {
                'detected': False,
                'confidence': 'low',
                'keywords_found': [],
                'markers_found': [],
                'findings_found': [],
                'threshold_violations': []
            }
            
            # Check keywords
            if 'keywords' in disease_data:
                for keyword in disease_data['keywords']:
                    if keyword in text_lower:
                        disease_analysis['keywords_found'].append(keyword)
            
            # Check markers
            if 'markers' in disease_data:
                for marker in disease_data['markers']:
                    if marker.lower() in text_lower:
                        disease_analysis['markers_found'].append(marker)
            
            # Check findings
            if 'findings' in disease_data:
                for finding in disease_data['findings']:
                    if finding in text_lower:
                        disease_analysis['findings_found'].append(finding)
            
            # Check threshold violations
            violations = self.check_threshold_violations(clinical_values, disease_type)
            disease_analysis['threshold_violations'] = violations
            
            # Calculate detection and confidence
            total_indicators = (
                len(disease_analysis['keywords_found']) +
                len(disease_analysis['markers_found']) +
                len(disease_analysis['findings_found']) +
                len(disease_analysis['threshold_violations'])
            )
            
            if total_indicators > 0:
                disease_analysis['detected'] = True
                
                if total_indicators >= 4 or len(disease_analysis['threshold_violations']) >= 2:
                    disease_analysis['confidence'] = 'high'
                elif total_indicators >= 2:
                    disease_analysis['confidence'] = 'medium'
                else:
                    disease_analysis['confidence'] = 'low'
            
            # Check for negative indicators specific to this disease
            negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
            if negative_count >= 2 and total_indicators < 3:
                disease_analysis['detected'] = False
            
            if disease_analysis['detected']:
                analysis['diseases_detected'].append(disease_type)
                analysis['overall_status'] = 'positive'
            
            analysis['all_findings'][disease_type] = disease_analysis
        
        # Determine overall confidence
        if analysis['diseases_detected']:
            max_confidence = max([
                analysis['all_findings'][d]['confidence'] 
                for d in analysis['diseases_detected']
            ], key=lambda x: ['low', 'medium', 'high'].index(x))
            analysis['confidence'] = max_confidence
        
        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_recommendations(self, analysis):
        """Generate disease-specific recommendations"""
        recommendations = []
        
        disease_recommendations = {
            'diabetes': [
                "Monitor blood glucose levels regularly",
                "Consult endocrinologist for diabetes management",
                "Consider HbA1c test every 3 months",
                "Maintain healthy diet and exercise routine"
            ],
            'heart_disease': [
                "Immediate cardiology consultation recommended",
                "Consider ECG and echocardiogram if not done",
                "Monitor blood pressure regularly",
                "Discuss lipid management with physician"
            ],
            'liver_disease': [
                "Consult hepatologist or gastroenterologist",
                "Avoid alcohol and hepatotoxic medications",
                "Consider liver function test monitoring",
                "Ultrasound or fibroscan may be needed"
            ],
            'kidney_disease': [
                "Consult nephrologist immediately",
                "Monitor kidney function regularly (eGFR, creatinine)",
                "Control blood pressure and blood sugar",
                "Adjust medications for kidney function"
            ],
            'cancer': [
                "Urgent oncology consultation required",
                "Further imaging and biopsy may be needed",
                "Discuss staging and treatment options",
                "Consider second opinion from cancer specialist"
            ],
            'breast_cancer': [
                "Immediate oncology and breast surgery consultation",
                "Mammogram and possible biopsy needed",
                "Discuss treatment options (surgery, chemo, radiation)",
                "Genetic counseling may be recommended"
            ],
            'hepatitis': [
                "Consult infectious disease specialist or hepatologist",
                "Determine hepatitis type and viral load",
                "Discuss antiviral treatment options",
                "Monitor liver function regularly"
            ]
        }
        
        if analysis['overall_status'] == 'positive':
            recommendations.append("⚠️ URGENT: Consult with healthcare professional immediately")
            
            for disease in analysis['diseases_detected']:
                if disease in disease_recommendations:
                    recommendations.extend(disease_recommendations[disease])
            
            recommendations.append("Bring original report and all previous medical records")
            recommendations.append("Do not delay medical consultation")
        else:
            recommendations.extend([
                "Continue regular health checkups",
                "Maintain healthy lifestyle",
                "Keep monitoring health parameters",
                "Follow preventive care guidelines"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_report(self, analysis, extracted_text, output_path="analysis_report.txt"):
        """Generate comprehensive multi-disease analysis report"""
        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE MEDICAL REPORT ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        # Overall Status
        status_symbol = "⚠️ WARNING" if analysis['overall_status'] == 'positive' else "✓ CLEAR"
        report.append(f"Overall Status: {status_symbol}")
        report.append(f"Confidence Level: {analysis['confidence'].upper()}")
        report.append("")
        
        # Diseases Detected
        if analysis['diseases_detected']:
            report.append("=" * 70)
            report.append("DISEASES/CONDITIONS DETECTED:")
            report.append("=" * 70)
            for disease in analysis['diseases_detected']:
                disease_name = disease.replace('_', ' ').title()
                confidence = analysis['all_findings'][disease]['confidence'].upper()
                report.append(f"\n⚠️ {disease_name} (Confidence: {confidence})")
                report.append("-" * 70)
                
                findings = analysis['all_findings'][disease]
                
                if findings['keywords_found']:
                    report.append("  Keywords Found:")
                    for kw in findings['keywords_found'][:5]:  # Limit to 5
                        report.append(f"    • {kw}")
                
                if findings['markers_found']:
                    report.append("  Biomarkers Mentioned:")
                    for marker in findings['markers_found']:
                        report.append(f"    • {marker}")
                
                if findings['threshold_violations']:
                    report.append("  ⚠️ ABNORMAL VALUES:")
                    for violation in findings['threshold_violations']:
                        report.append(f"    • {violation}")
                
                if findings['findings_found']:
                    report.append("  Clinical Findings:")
                    for finding in findings['findings_found'][:5]:
                        report.append(f"    • {finding}")
                
                report.append("")
        else:
            report.append("✓ No significant disease indicators detected")
            report.append("")
        
        # Clinical Values Extracted
        if analysis['clinical_values']:
            report.append("=" * 70)
            report.append("CLINICAL VALUES EXTRACTED:")
            report.append("=" * 70)
            for marker, value in analysis['clinical_values'].items():
                report.append(f"  • {marker}: {value}")
            report.append("")
        
        # Recommendations
        report.append("=" * 70)
        report.append("RECOMMENDATIONS:")
        report.append("=" * 70)
        for i, rec in enumerate(analysis['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Disclaimer
        report.append("=" * 70)
        report.append("DISCLAIMER:")
        report.append("=" * 70)
        report.append("This is an automated analysis tool and NOT a substitute for")
        report.append("professional medical advice. Always consult qualified healthcare")
        report.append("professionals for proper diagnosis and treatment.")
        report.append("")
        
        report.append("=" * 70)
        report.append("EXTRACTED TEXT SAMPLE (First 500 characters):")
        report.append("=" * 70)
        report.append(extracted_text[:500] + "...")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def analyze_report(self, file_path, output_dir="analysis_output"):
        """Main method to analyze medical report for multiple diseases"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📄 Processing: {file_path}")
        
        # Determine file type and extract text
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            print("🔄 Converting PDF to images and extracting text...")
            extracted_text = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            print("🔄 Extracting text from image...")
            extracted_text = self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Save extracted text
        text_file = os.path.join(output_dir, "extracted_text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"✓ Text extracted and saved to {text_file}")
        
        # Analyze for all diseases
        print("🔍 Analyzing for multiple disease indicators...")
        analysis = self.analyze_disease_indicators(extracted_text)
        
        # Generate report
        report_file = os.path.join(output_dir, "comprehensive_analysis_report.txt")
        report = self.generate_report(analysis, extracted_text, report_file)
        
        print("\n" + report)
        print(f"\n✓ Complete analysis saved to {report_file}")
        
        # Save JSON results
        json_file = os.path.join(output_dir, "analysis_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ JSON results saved to {json_file}")
        
        return analysis, extracted_text


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("MULTI-DISEASE MEDICAL REPORT ANALYZER")
    print("="*70)
    print("\nSupported Disease Detection:")
    print("  • Cancer (Lung, Breast, General)")
    print("  • Diabetes (Type 1 & 2)")
    print("  • Heart Disease")
    print("  • Liver Disease")
    print("  • Chronic Kidney Disease")
    print("  • Hepatitis (A, B, C)")
    print("="*70)
    print()
    
    # Initialize analyzer
    analyzer = MedicalReportAnalyzer()
    
    # Analyze the report
    file_path = "PE02610500081224678_RLS (1).pdf"  # Can be .png, .jpg, or .pdf
    
    try:
        analysis, text = analyzer.analyze_report(file_path)
        
        # Quick summary
        print("\n" + "="*70)
        print("QUICK SUMMARY")
        print("="*70)
        
        if analysis['overall_status'] == 'positive':
            print("Status: DISEASE INDICATORS DETECTED")
            print(f"Confidence: {analysis['confidence'].upper()}")
            print(f"\nDiseases Detected: {len(analysis['diseases_detected'])}")
            for disease in analysis['diseases_detected']:
                disease_name = disease.replace('_', ' ').title()
                conf = analysis['all_findings'][disease]['confidence']
                print(f"  • {disease_name} ({conf} confidence)")
        else:
            print("✓ Status: NO SIGNIFICANT DISEASE INDICATORS")
            print("   All parameters appear normal")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n Come again!:")
    