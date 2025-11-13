import cv2
import numpy as np
import pytesseract
import os
import re
from pdf2image import convert_from_path
from pathlib import Path
import json

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class MedicalReportAnalyzer:
    def __init__(self, poppler_path=None):

        self.poppler_path = poppler_path
        
        # Disease indicators - keywords that suggest positive findings
        self.disease_keywords = {
            'cancer': ['cancer', 'malignancy', 'malignant', 'tumor', 'neoplasm', 
                      'carcinoma', 'metastasis', 'metastatic', 'suspicious nodules'],
            'positive_findings': ['positive', 'detected', 'abnormal', 'elevated', 
                                 'outside reference range', 'significant'],
            'severity': ['severe', 'acute', 'critical', 'high risk', 'advanced'],
            'infections': ['infection', 'bacterial', 'viral', 'pathogen', 'sepsis'],
            'inflammation': ['inflammation', 'inflammatory', 'inflamed']
        }
        
        # Negative indicators - keywords that suggest no disease
        self.negative_keywords = [
            'negative', 'normal', 'no evidence', 'no such symptoms',
            'no trace', 'within normal limits', 'unremarkable',
            'no abnormality', 'healthy', 'clear', 'no suspicious'
        ]
        
        # High-priority clinical sections
        self.clinical_sections = [
            'outcome', 'conclusions', 'findings', 'results', 
            'diagnosis', 'impression', 'summary'
        ]
    
    def preprocess_image(self, image):
        """Enhance image quality for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Adaptive thresholding for better text extraction
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
        
        # Preprocess
        processed = self.preprocess_image(img)
        
        # Extract text with better configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config, lang='eng')
        
        return text
    
    def extract_text_from_pdf(self, pdf_path, output_dir="temp_images"):
        """Convert PDF to images and extract text"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Use self.poppler_path from initialization
        if self.poppler_path:
            images = convert_from_path(pdf_path, dpi=300, poppler_path=self.poppler_path)
        else:
            try:
                images = convert_from_path(pdf_path, dpi=300)
            except Exception as e:
                raise Exception(
                    f"Poppler not found. Initialize with poppler_path parameter.\n"
                    f"Example: analyzer = MedicalReportAnalyzer(poppler_path=r'C:\Program Files\poppler\Library\bin')"
                )
        
        full_text = ""
        for i, image in enumerate(images):
            temp_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(temp_path, 'PNG')
            text = self.extract_text_from_image(temp_path)
            full_text += f"\n--- Page {i+1} ---\n{text}"
        
        return full_text

    def clean_text(self, text):
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase for analysis
        return text.strip()
    
    def extract_key_values(self, text):
        """Extract important clinical values and markers"""
        results = {}
        
        # Extract tumor markers with values
        tumor_marker_pattern = r'(CA\s*\d+-?\d*|CEA|NSE|ProGRP|SCC|AFP|PSA)\s*[:\(]?\s*([^\)]+\)?)?\s*:?\s*(\d+\.?\d*)'
        matches = re.findall(tumor_marker_pattern, text, re.IGNORECASE)
        
        if matches:
            results['tumor_markers'] = {}
            for marker, unit, value in matches:
                marker = marker.strip()
                results['tumor_markers'][marker] = float(value)
        
        return results
    
    def analyze_disease_indicators(self, text):
        """Analyze text for disease indicators"""
        text_lower = text.lower()
        
        analysis = {
            'disease_detected': False,
            'confidence': 'low',
            'disease_types': [],
            'positive_indicators': [],
            'negative_indicators': [],
            'critical_findings': [],
            'tumor_markers_abnormal': False
        }
        
        # Check for negative indicators first
        for keyword in self.negative_keywords:
            if keyword in text_lower:
                analysis['negative_indicators'].append(keyword)
        
        # Check for disease keywords
        for disease_type, keywords in self.disease_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                analysis['disease_types'].append(disease_type)
                analysis['positive_indicators'].extend(found_keywords)
        
        # Extract clinical values
        clinical_data = self.extract_key_values(text)
        
        # Check tumor markers
        if 'tumor_markers' in clinical_data:
            analysis['tumor_markers_abnormal'] = True
            analysis['disease_types'].append('abnormal_markers')
        
        # Look for critical sections
        for section in self.clinical_sections:
            section_pattern = rf'{section}[:\s]+([^\n]+)'
            matches = re.findall(section_pattern, text_lower)
            if matches:
                for match in matches:
                    if any(kw in match for kw in ['malignancy', 'cancer', 'positive', 'abnormal']):
                        analysis['critical_findings'].append(match.strip())
        
        # Determine if disease is detected
        positive_score = len(analysis['positive_indicators'])
        negative_score = len(analysis['negative_indicators'])
        
        if positive_score > 0:
            analysis['disease_detected'] = True
            
            # Determine confidence
            if positive_score >= 3 or analysis['tumor_markers_abnormal']:
                analysis['confidence'] = 'high'
            elif positive_score >= 2:
                analysis['confidence'] = 'medium'
            else:
                analysis['confidence'] = 'low'
        
        # If strong negative indicators, override
        if negative_score >= 2 and 'negative cancer' in text_lower:
            analysis['disease_detected'] = False
            analysis['confidence'] = 'high'
        
        return analysis
    
    def generate_report(self, analysis, extracted_text, output_path="analysis_report.txt"):
        """Generate a detailed analysis report"""
        report = []
        report.append("=" * 60)
        report.append("MEDICAL REPORT ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Disease Detection Status
        status = "⚠️ DISEASE DETECTED" if analysis['disease_detected'] else "✓ NO DISEASE DETECTED"
        report.append(f"Status: {status}")
        report.append(f"Confidence: {analysis['confidence'].upper()}")
        report.append("")
        
        # Disease Types
        if analysis['disease_types']:
            report.append("Disease Categories Found:")
            for dtype in set(analysis['disease_types']):
                report.append(f"  • {dtype.replace('_', ' ').title()}")
            report.append("")
        
        # Positive Indicators
        if analysis['positive_indicators']:
            report.append("Positive Indicators Found:")
            for indicator in set(analysis['positive_indicators']):
                report.append(f"  • {indicator}")
            report.append("")
        
        # Negative Indicators
        if analysis['negative_indicators']:
            report.append("Negative Indicators Found:")
            for indicator in set(analysis['negative_indicators']):
                report.append(f"  • {indicator}")
            report.append("")
        
        # Critical Findings
        if analysis['critical_findings']:
            report.append("Critical Findings in Clinical Sections:")
            for finding in analysis['critical_findings']:
                report.append(f"  • {finding}")
            report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        if analysis['disease_detected']:
            report.append("  • Consult with a healthcare professional immediately")
            report.append("  • Follow up with additional tests as recommended")
            report.append("  • Bring original report to medical appointment")
        else:
            report.append("  • Continue regular health checkups")
            report.append("  • Maintain healthy lifestyle")
        report.append("")
        
        report.append("=" * 60)
        report.append("EXTRACTED TEXT SAMPLE (First 500 chars)")
        report.append("=" * 60)
        report.append(extracted_text[:500] + "...")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def analyze_report(self, file_path, output_dir="analysis_output"):
        """Main method to analyze a medical report"""
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
        
        # Analyze for disease indicators
        print("🔍 Analyzing for disease indicators...")
        analysis = self.analyze_disease_indicators(extracted_text)
        
        # Generate report
        report_file = os.path.join(output_dir, "analysis_report.txt")
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
    # Initialize analyzer
    analyzer = MedicalReportAnalyzer()
    
    # Analyze the report
    file_path = "lung_sample.png"  # Can be .png, .jpg, or .pdf
    
    try:
        analysis, text = analyzer.analyze_report(file_path)
        
        # Quick summary
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        print(f"Disease Detected: {'YES ⚠️' if analysis['disease_detected'] else 'NO ✓'}")
        print(f"Confidence Level: {analysis['confidence'].upper()}")
        print(f"Disease Types: {', '.join(set(analysis['disease_types'])) if analysis['disease_types'] else 'None'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()