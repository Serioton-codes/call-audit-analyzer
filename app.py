import streamlit as st
import sqlite3
import hashlib
import datetime
import os
import tempfile
from datetime import date, datetime, timedelta
import requests
import json
import time
import re
import whisper 
import pandas as pd
from io import BytesIO
import base64
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Call Audit Analyzer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Audit Parameters Configuration (Based on your requirements)
AUDIT_PARAMETERS = {
    "authentication_demographics": {
        "weight": 1,
        "description": "Authentication/Demographics related questions",
        "criteria": ["Name verification", "Date of birth confirmation", "Address verification"],
        "keywords": ["name", "birth", "address", "verify", "confirm", "authentication", "demographics"]
    },
    "agent_probing_statements": {
        "weight": 2,
        "description": "Agent probing questions and required statements",
        "criteria": ["Required disclosures", "Probing questions", "Compliance statements"],
        "keywords": ["disclosure", "understand", "agree", "consent", "authorize", "probing", "required"]
    },
    "member_understanding": {
        "weight": 3,
        "description": "Member understanding related questions",
        "criteria": ["Explanation clarity", "Member comprehension", "Question answering"],
        "keywords": ["understand", "clear", "questions", "explain", "benefits", "comprehension"]
    },
    "product_explanation": {
        "weight": 4,
        "description": "Product explanation and PI safety",
        "criteria": ["Product features explained", "Benefits outlined", "Safety information"],
        "keywords": ["benefits", "coverage", "features", "plan", "protection", "product", "safety"]
    },
    "ethical_practice": {
        "weight": 8,
        "description": "Ethical agent practice",
        "criteria": ["No misleading statements", "Honest representation", "No pressure tactics"],
        "keywords": ["honest", "truthful", "ethical", "transparent", "clear"]
    }
}

# Database setup
def init_database():
    """Initialize SQLite database for call audit management"""
    conn = sqlite3.connect('call_audit.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            date_of_birth DATE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create call_audits table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS call_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            file_size INTEGER,
            transcription TEXT,
            audit_scores TEXT,
            overall_score REAL,
            pass_fail TEXT,
            analysis_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create convoso_integration table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS convoso_integration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            api_key TEXT,
            endpoint_url TEXT,
            last_sync TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(first_name, last_name, email, dob, password):
    """Create new user in database"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute('''
            INSERT INTO users (first_name, last_name, email, date_of_birth, password_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (first_name, last_name, email, dob, password_hash))
        
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already exists!"
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def authenticate_user(email, password):
    """Authenticate user login"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute('''
            SELECT id, first_name, last_name, email FROM users 
            WHERE email = ? AND password_hash = ?
        ''', (email, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {
                'id': user[0],
                'first_name': user[1],
                'last_name': user[2],
                'email': user[3]
            }
        else:
            return False, "Invalid email or password"
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

@st.cache_resource
def load_whisper_model():
    """Load Whisper model for transcription (cached)"""
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

def transcribe_audio_whisper(audio_file):
    """Transcribe audio using OpenAI Whisper (free)"""
    try:
        model = load_whisper_model()
        if model is None:
            return None
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Transcribe
        result = model.transcribe(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def analyze_call_with_ai(transcription):
    """Analyze call transcription using advanced rule-based + NLP approach"""
    analysis = {
        "parameter_scores": {},
        "detailed_analysis": {},
        "overall_score": 0,
        "pass_fail": "FAIL",
        "recommendations": []
    }
    
    try:
        transcription_lower = transcription.lower()
        total_weighted_score = 0
        max_possible_score = 0
        
        for param_name, param_config in AUDIT_PARAMETERS.items():
            weight = param_config["weight"]
            keywords = param_config["keywords"]
            criteria = param_config["criteria"]
            
            # Advanced scoring algorithm
            parameter_score = calculate_parameter_score(transcription_lower, param_name, param_config)
            
            analysis["parameter_scores"][param_name] = {
                "score": round(parameter_score, 2),
                "weight": weight,
                "weighted_score": round(parameter_score * weight / 100, 2),
                "assessment": get_parameter_assessment(parameter_score)
            }
            
            analysis["detailed_analysis"][param_name] = {
                "description": param_config["description"],
                "keywords_found": [kw for kw in keywords if kw in transcription_lower],
                "assessment": get_parameter_assessment(parameter_score),
                "specific_feedback": get_specific_feedback(param_name, parameter_score, transcription_lower)
            }
            
            total_weighted_score += parameter_score * weight / 100
            max_possible_score += weight
        
        # Calculate overall score
        overall_score = (total_weighted_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        analysis["overall_score"] = round(overall_score, 2)
        analysis["pass_fail"] = "PASS" if overall_score >= 85 else "FAIL"
        
        # Generate comprehensive recommendations
        analysis["recommendations"] = generate_comprehensive_recommendations(analysis["parameter_scores"], overall_score)
        
        return analysis
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return analysis

def calculate_parameter_score(transcription, param_name, param_config):
    """Advanced scoring algorithm for each parameter"""
    keywords = param_config["keywords"]
    weight = param_config["weight"]
    
    # Base scoring components
    keyword_score = 0
    context_score = 0
    pattern_score = 0
    sentiment_score = 0
    
    # 1. Keyword matching with proximity scoring
    keyword_matches = 0
    for keyword in keywords:
        if keyword in transcription:
            keyword_matches += 1
            # Bonus for multiple occurrences
            keyword_count = transcription.count(keyword)
            keyword_score += min(15, keyword_count * 5)
    
    # 2. Context-specific pattern analysis
    if param_name == "authentication_demographics":
        patterns = [
            "can you verify", "confirm your", "what is your", "date of birth",
            "full name", "address", "social security", "member id"
        ]
        context_score = analyze_patterns(transcription, patterns, base_score=40)
        
    elif param_name == "agent_probing_statements":
        patterns = [
            "do you understand", "are you interested", "would you like",
            "can i explain", "may i ask", "let me tell you", "disclosure"
        ]
        context_score = analyze_patterns(transcription, patterns, base_score=35)
        
    elif param_name == "member_understanding":
        patterns = [
            "do you have questions", "is that clear", "understand so far",
            "make sense", "any questions", "follow me", "does that sound"
        ]
        context_score = analyze_patterns(transcription, patterns, base_score=30)
        
    elif param_name == "product_explanation":
        patterns = [
            "this plan includes", "benefits are", "coverage includes",
            "plan offers", "you will receive", "this covers", "deductible"
        ]
        context_score = analyze_patterns(transcription, patterns, base_score=25)
        
    elif param_name == "ethical_practice":
        # For ethical practice, check for negative indicators
        negative_patterns = [
            "you have to", "must decide now", "limited time", "pressure",
            "only today", "special deal", "act fast", "won't get another chance"
        ]
        negative_score = analyze_patterns(transcription, negative_patterns, base_score=0)
        context_score = max(0, 85 - negative_score)  # Start high, deduct for bad practices
    
    # 3. Length and completeness analysis
    if len(transcription.split()) > 50:  # Reasonable conversation length
        pattern_score += 10
    
    # 4. Professional language detection
    professional_indicators = [
        "thank you", "please", "may i", "would you", "i understand",
        "let me help", "i appreciate", "thank you for"
    ]
    for indicator in professional_indicators:
        if indicator in transcription:
            sentiment_score += 5
    
    # Combine scores with weights
    final_score = (
        keyword_score * 0.3 +
        context_score * 0.4 +
        pattern_score * 0.2 +
        min(sentiment_score, 20) * 0.1
    )
    
    return min(100, max(0, final_score))

def analyze_patterns(transcription, patterns, base_score=30):
    """Analyze specific patterns in transcription"""
    matches = sum(1 for pattern in patterns if pattern in transcription)
    if matches == 0:
        return base_score
    return min(100, base_score + matches * 20)

def get_parameter_assessment(score):
    """Get qualitative assessment based on score"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Satisfactory"
    elif score >= 60:
        return "Needs Improvement"
    else:
        return "Poor"

def get_specific_feedback(param_name, score, transcription):
    """Generate specific feedback for each parameter"""
    feedback = []
    
    if param_name == "authentication_demographics":
        if score < 70:
            feedback.append("Ensure proper member authentication is completed")
            feedback.append("Verify all required demographic information")
        else:
            feedback.append("Good authentication process")
    
    elif param_name == "agent_probing_statements":
        if score < 70:
            feedback.append("Include more probing questions to understand member needs")
            feedback.append("Ensure all required disclosures are made")
        else:
            feedback.append("Effective probing and disclosure statements")
    
    elif param_name == "member_understanding":
        if score < 70:
            feedback.append("Check member understanding more frequently")
            feedback.append("Ask more clarifying questions")
        else:
            feedback.append("Good confirmation of member understanding")
    
    elif param_name == "product_explanation":
        if score < 70:
            feedback.append("Provide more detailed product explanations")
            feedback.append("Ensure all benefits and features are covered")
        else:
            feedback.append("Comprehensive product explanation provided")
    
    elif param_name == "ethical_practice":
        if score < 70:
            feedback.append("CRITICAL: Review for ethical concerns")
            feedback.append("Avoid pressure tactics and misleading statements")
        else:
            feedback.append("Ethical standards maintained")
    
    return feedback

def generate_comprehensive_recommendations(parameter_scores, overall_score):
    """Generate detailed improvement recommendations"""
    recommendations = []
    
    # Overall performance assessment
    if overall_score >= 95:
        recommendations.append("🌟 Outstanding performance! Continue maintaining these high standards.")
    elif overall_score >= 85:
        recommendations.append("✅ Good performance overall. Focus on minor improvements for excellence.")
    elif overall_score >= 70:
        recommendations.append("⚠️ Performance needs improvement. Focus on key areas below.")
    else:
        recommendations.append("🚨 URGENT: Significant improvement required across multiple areas.")
    
    # Parameter-specific recommendations
    priority_issues = []
    for param_name, scores in parameter_scores.items():
        param_desc = AUDIT_PARAMETERS[param_name]["description"]
        weight = AUDIT_PARAMETERS[param_name]["weight"]
        
        if scores["score"] < 60:
            priority_issues.append((param_name, scores["score"], weight))
            recommendations.append(f"🔴 HIGH PRIORITY: {param_desc} - Score: {scores['score']}%")
        elif scores["score"] < 80:
            recommendations.append(f"🟡 IMPROVE: {param_desc} - Score: {scores['score']}%")
    
    # Weight-based priority recommendations
    if priority_issues:
        priority_issues.sort(key=lambda x: x[2], reverse=True)  # Sort by weight
        top_priority = priority_issues[0]
        recommendations.append(f"🎯 TOP PRIORITY: Focus on {top_priority[0].replace('_', ' ').title()} (Weight: {top_priority[2]})")
    
    # Specific improvement actions
    if any(scores["score"] < 70 for scores in parameter_scores.values()):
        recommendations.extend([
            "📚 Recommended Actions:",
            "• Review call scripts and compliance requirements",
            "• Practice active listening and confirmation techniques",
            "• Ensure all required disclosures are included",
            "• Focus on clear, ethical communication"
        ])
    
    return recommendations

def save_audit(user_id, filename, file_size, transcription, analysis):
    """Save call audit to database"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO call_audits (user_id, filename, file_size, transcription, 
                                   audit_scores, overall_score, pass_fail, analysis_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, filename, file_size, transcription, 
              json.dumps(analysis["parameter_scores"]), 
              analysis["overall_score"], analysis["pass_fail"],
              json.dumps(analysis)))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving audit: {str(e)}")
        return False

def get_user_audits(user_id):
    """Get all audits for a user"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_size, transcription, audit_scores, 
                   overall_score, pass_fail, analysis_details, created_at
            FROM call_audits 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        audits = cursor.fetchall()
        conn.close()
        return audits
    except Exception as e:
        st.error(f"Error fetching audits: {str(e)}")
        return []

def delete_audit(audit_id, user_id):
    """Delete a specific audit record"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM call_audits 
            WHERE id = ? AND user_id = ?
        ''', (audit_id, user_id))
        
        deleted_rows = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_rows > 0:
            return True, "Audit deleted successfully!"
        else:
            return False, "Audit not found or unauthorized"
    except Exception as e:
        return False, f"Error deleting audit: {str(e)}"

def delete_all_user_audits(user_id):
    """Delete all audits for a specific user"""
    try:
        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM call_audits 
            WHERE user_id = ?
        ''', (user_id,))
        
        deleted_rows = cursor.rowcount
        conn.commit()
        conn.close()
        
        return True, f"Successfully deleted {deleted_rows} audit(s)!"
    except Exception as e:
        return False, f"Error deleting audits: {str(e)}"

def export_audit_report(audit_data):
    """Export audit report to Excel format"""
    try:
        analysis = json.loads(audit_data[7])  # analysis_details
        parameter_scores = json.loads(audit_data[4])  # audit_scores
        
        # Create DataFrame for export
        report_data = []
        for param_name, scores in parameter_scores.items():
            report_data.append({
                'Parameter': AUDIT_PARAMETERS[param_name]['description'],
                'Score (%)': scores['score'],
                'Weight': scores['weight'],
                'Weighted Score': scores['weighted_score'],
                'Assessment': scores['assessment']
            })
        
        df = pd.DataFrame(report_data)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Filename', 'Overall Score', 'Pass/Fail', 'Date'],
                'Value': [audit_data[1], f"{audit_data[5]}%", audit_data[6], audit_data[8][:16]]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed scores
            df.to_excel(writer, sheet_name='Detailed Scores', index=False)
            
            # Recommendations
            rec_df = pd.DataFrame({'Recommendations': analysis['recommendations']})
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")
        return None

# Initialize database
init_database()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# Sidebar navigation
with st.sidebar:
    st.title("📞 Call Audit Analyzer")
    st.markdown("*AI-Powered Quality Assessment*")
    
    if st.session_state.logged_in:
        st.success(f"Welcome, {st.session_state.user['first_name']}!")
        
        menu = st.radio("Navigation", [
            "🏠 Dashboard", 
            "📤 Upload Calls", 
            "📊 Audit History", 
            "⚙️ Settings",
            "🚪 Logout"
        ])
        
        if menu == "🚪 Logout":
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()
        elif menu == "🏠 Dashboard":
            st.session_state.page = 'dashboard'
        elif menu == "📤 Upload Calls":
            st.session_state.page = 'upload'
        elif menu == "📊 Audit History":
            st.session_state.page = 'audits'
        elif menu == "⚙️ Settings":
            st.session_state.page = 'settings'
    else:
        st.info("Please log in to continue")
        auth_option = st.radio("Choose an option:", ["🔐 Login", "📝 Sign Up"])
        st.session_state.page = 'login' if auth_option == "🔐 Login" else 'signup'

# Main content area
if not st.session_state.logged_in:
    if st.session_state.page == 'signup':
        st.title("📝 Create Account")
        st.markdown("Join the Call Audit Platform")
        
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name*", placeholder="John")
                last_name = st.text_input("Last Name*", placeholder="Doe")
            
            with col2:
                email = st.text_input("Email*", placeholder="john.doe@example.com")
                dob = st.date_input("Date of Birth*", 
                                   max_value=date.today(),
                                   min_value=date(1900, 1, 1))
            
            password = st.text_input("Password*", type="password", 
                                   help="Minimum 6 characters")
            confirm_password = st.text_input("Confirm Password*", type="password")
            
            terms = st.checkbox("I agree to the Terms of Service and Privacy Policy*")
            
            submit = st.form_submit_button("Create Account", type="primary")
            
            if submit:
                if not all([first_name, last_name, email, password, confirm_password]):
                    st.error("Please fill in all required fields")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif not terms:
                    st.error("Please accept the terms and conditions")
                else:
                    success, message = create_user(first_name, last_name, email, dob, password)
                    if success:
                        st.success(message)
                        st.info("Please log in with your new account")
                        time.sleep(2)
                        st.session_state.page = 'login'
                        st.rerun()
                    else:
                        st.error(message)
    
    else:  # Login page
        st.title("🔐 Login")
        st.markdown("Welcome to Call Audit Analyzer")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            
            remember = st.checkbox("Remember me")
            
            submit = st.form_submit_button("Sign In", type="primary")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    success, result = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = result
                        st.session_state.page = 'dashboard'
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(result)

else:  # User is logged in
    if st.session_state.page == 'dashboard':
        st.title(f"🏠 Dashboard - Welcome, {st.session_state.user['first_name']}!")
        
        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        audits = get_user_audits(st.session_state.user['id'])
        
        with col1:
            st.metric("Total Audits", len(audits))
        
        with col2:
            pass_count = len([a for a in audits if a[6] == "PASS"])
            pass_rate = (pass_count / len(audits) * 100) if audits else 0
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        with col3:
            avg_score = sum([a[5] for a in audits]) / len(audits) if audits else 0
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col4:
            recent_audits = len([a for a in audits[:5]])  # Last 5 audits
            st.metric("Recent Audits", recent_audits)
        
        # Quick Stats Chart
        if audits:
            st.subheader("📈 Performance Trend")
            
            # Create trend data
            trend_data = []
            for audit in audits[-10:]:  # Last 10 audits
                trend_data.append({
                    'Audit': audit[1][:15] + "..." if len(audit[1]) > 15 else audit[1],
                    'Score': audit[5],
                    'Status': audit[6]
                })
            
            trend_df = pd.DataFrame(trend_data)
            if not trend_df.empty:
                st.bar_chart(trend_df.set_index('Audit')['Score'])
        
        # Audit Parameters Overview
        st.subheader("📋 Audit Parameters & Weights")
        st.markdown("*Higher weights indicate greater risk to member and plan*")
        
        param_df = pd.DataFrame([
            {
                'Parameter': config['description'],
                'Weight': config['weight'],
                'Risk Level': 'Critical' if config['weight'] >= 8 else 'High' if config['weight'] >= 4 else 'Medium' if config['weight'] >= 2 else 'Low',
                'Impact': 'Ethical violations' if config['weight'] >= 8 else 'Member complaints' if config['weight'] >= 4 else 'Compliance issues' if config['weight'] >= 2 else 'Process issues'
            }
            for config in AUDIT_PARAMETERS.values()
        ])
        st.dataframe(param_df, use_container_width=True)
        
        # Recent audits preview
        st.subheader("📊 Recent Audits")
        if audits:
            for audit in audits[:3]:  # Show last 3
                score_color = "🟢" if audit[6] == "PASS" else "🔴"
                with st.expander(f"{score_color} {audit[1]} - Score: {audit[5]}% ({audit[6]}) - {audit[8][:16]}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**File Size:** {audit[2]:,} bytes")
                        st.write(f"**Overall Score:** {audit[5]}%")
                        st.write(f"**Result:** {audit[6]}")
                        
                        # Show parameter breakdown
                        if audit[4]:  # audit_scores
                            try:
                                scores = json.loads(audit[4])
                                st.write("**Top Issues:**")
                                sorted_params = sorted(scores.items(), key=lambda x: x[1]['score'])
                                for param_name, param_scores in sorted_params[:2]:
                                    param_desc = AUDIT_PARAMETERS[param_name]['description']
                                    st.write(f"• {param_desc}: {param_scores['score']}%")
                            except:
                                pass
                    
                    with col_b:
                        st.write("**Transcription Preview:**")
                        preview_text = audit[3][:300] + "..." if len(audit[3]) > 300 else audit[3]
                        st.write(preview_text)
        else:
            st.info("🚀 No audits yet. Upload your first call recording to get started!")
            if st.button("📤 Upload Call Recording", type="primary"):
                st.session_state.page = 'upload'
                st.rerun()
    
    elif st.session_state.page == 'upload':
        st.title("📤 Upload Call Recordings")
        st.markdown("Upload audio files for automated AI-powered quality audit")
        
        # Information panel
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Automated Call Audit Process:**
            
            1. **Upload**: Select call recordings (MP3, WAV, M4A, MP4)
            2. **Transcription**: AI transcribes audio using Whisper (free)
            3. **Analysis**: Advanced AI evaluates based on 5 key parameters
            4. **Scoring**: Weighted scoring system (1-8 points per parameter)
                        5. **Results**: Pass/Fail with detailed parameter scores, overall score, and recommendations
            6. **History**: Every audit is stored in your dashboard for later export (Excel) or deletion
            """)

        # -------- Upload Handler --------
        uploaded_files = st.file_uploader(
            "Select one or more audio files",
            type=['mp3', 'wav', 'm4a', 'mp4'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
            for idx, file in enumerate(uploaded_files):

                with st.expander(f"📂 {file.name}"):
                    st.audio(file, format="audio/mp3")

                    if st.button("🔍 Run Audit", key=f"audit_{idx}"):

                        with st.spinner("Transcribing… this might take a minute ⏳"):
                            transcript_text = transcribe_audio_whisper(file)

                        if transcript_text:
                            with st.spinner("Analyzing call quality…"):
                                analysis_result = analyze_call_with_ai(transcript_text)

                            save_ok = save_audit(
                                user_id=st.session_state.user['id'],
                                filename=file.name,
                                file_size=file.size,
                                transcription=transcript_text,
                                analysis=analysis_result
                            )

                            if save_ok:
                                st.success(f"🎉 Audit complete — overall score **{analysis_result['overall_score']}%** ({analysis_result['pass_fail']})")
                                
                                # Show quick metrics
                                colA, colB = st.columns(2)
                                with colA:
                                    st.metric("Overall Score", f"{analysis_result['overall_score']}%")
                                    st.metric("Result", analysis_result['pass_fail'])
                                with colB:
                                    st.write("### Parameter Breakdown")
                                    for p, s in analysis_result['parameter_scores'].items():
                                        st.write(f"• **{AUDIT_PARAMETERS[p]['description']}** — {s['score']}%  ({s['assessment']})")

                                with st.expander("📑 Full Transcript"):
                                    st.text(transcript_text)

                                with st.expander("💡 Recommendations"):
                                    for rec in analysis_result['recommendations']:
                                        st.write(f"- {rec}")

                                st.balloons()
                            else:
                                st.error("Could not save audit to database.")

    # ------------------ Audit History Page ------------------
    elif st.session_state.page == 'audits':
        st.title("📊 Audit History")
        audits = get_user_audits(st.session_state.user['id'])

        if not audits:
            st.info("No audits found. Upload a call to see results here.")
        else:
            for audit in audits:
                audit_id, fname, fsize, trans, scores_json, overall, pf, details_json, created = audit
                score_tag = "🟢" if pf == "PASS" else "🔴"

                with st.expander(f"{score_tag} {fname} — {overall}% ({pf}) — {created[:16]}"):
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Size:** {fsize:,} bytes")
                        st.write(f"**Overall:** {overall}%")
                        st.write(f"**Result:** {pf}")
                        
                        if st.button("⬇️ Download Report (Excel)", key=f"dload_{audit_id}"):
                            excel_bytes = export_audit_report(audit)
                            if excel_bytes:
                                b64 = base64.b64encode(excel_bytes).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{fname}_audit.xlsx">👉 Click to download</a>'
                                st.markdown(href, unsafe_allow_html=True)

                        if st.button("🗑️ Delete Audit", key=f"del_{audit_id}"):
                            ok, msg = delete_audit(audit_id, st.session_state.user['id'])
                            st.toast(msg)
                            st.experimental_rerun()

                    with col2:
                        st.write("##### Parameter Scores")
                        try:
                            param_scores = json.loads(scores_json)
                            for p, s in param_scores.items():
                                st.write(f"- **{AUDIT_PARAMETERS[p]['description']}**: {s['score']}% ({s['assessment']})")
                        except:
                            st.write("Could not parse scores.")

                        with st.expander("📝 Transcript"):
                            st.text(trans)

                        with st.expander("📋 Full Analysis JSON"):
                            st.json(json.loads(details_json))

            # Mass‑delete option
            st.divider()
            if st.button("⚠️ Delete ALL my audits"):
                ok, msg = delete_all_user_audits(st.session_state.user['id'])
                st.toast(msg)
                st.experimental_rerun()

    # ------------------ Settings Page ------------------
    elif st.session_state.page == 'settings':
        st.title("⚙️ Settings")

        conn = sqlite3.connect('call_audit.db')
        cursor = conn.cursor()

        # Fetch existing integration (if any)
        cursor.execute(
            "SELECT id, api_key, endpoint_url FROM convoso_integration WHERE user_id = ?",
            (st.session_state.user['id'],)
        )
        record = cursor.fetchone()

        api_key = "" if record is None else record[1]
        endpoint = "" if record is None else record[2]

        st.subheader("🔌 Convoso Integration")
        with st.form("convoso_form"):
            new_api_key = st.text_input("API Key", value=api_key, type="password")
            new_endpoint = st.text_input("Endpoint URL", value=endpoint, placeholder="https://api.convoso.com/…")

            submitted = st.form_submit_button("Save")
            if submitted:
                if record is None:
                    # Insert
                    cursor.execute("""
                        INSERT INTO convoso_integration (user_id, api_key, endpoint_url, last_sync)
                        VALUES (?, ?, ?, ?)
                    """, (st.session_state.user['id'], new_api_key, new_endpoint, datetime.utcnow()))
                else:
                    # Update
                    cursor.execute("""
                        UPDATE convoso_integration
                        SET api_key = ?, endpoint_url = ?, last_sync = ?
                        WHERE id = ? AND user_id = ?
                    """, (new_api_key, new_endpoint, datetime.utcnow(), record[0], st.session_state.user['id']))

                conn.commit()
                st.success("Settings saved!")

        conn.close()

# ------------------  Footer ------------------
st.markdown("---")
st.caption("© 2024 Call Audit Analyzer | Built with ❤️ & Streamlit")
