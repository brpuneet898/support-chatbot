from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
from authlib.integrations.flask_client import OAuth

# Imports from other scripts
from vector_store import load_vector_store, create_vector_store_from_docs
from ask_llm import query_model_with_rag

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")  # secret key to secure cookies and session data.
oauth = OAuth(app) # OAuth is a way to safely let users login using Google without handling their passwords yourself

google = oauth.register( # Then you told OAuth: Hey OAuth
    
    name='google', # register Google as a login provider, and here’s my
    client_id= os.environ.get("GOOGLE_AUTH_CLIENT_ID"), # client_id
    client_secret= os.environ.get("GOOGLE_AUTH_CLIENT_SECRET"), # a secret password only your app and Google know
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration', # Google OpenID configuration URL (this tells your app where to send users to login)
    client_kwargs={'scope': 'openid email profile'} # meaning what user info you want to access.
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def get_answer_from_llm():
    try:
        query = request.get_json().get("message")
        if not query.strip():
            return jsonify({"response": "Please provide a valid query."}), 400
        answer = query_model_with_rag(query, vector_store, grading_doc_chunks)
        print(f"Query: {query}\nAnswer: {answer}")
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Response was not recieved because of an technical issue. Please contact support if issue persists."})

@app.route('/login')
def login():
    redirect_uri = url_for('authorized', _external=True)
    return google.authorize_redirect(redirect_uri) #User clicks Login → /login → Google login page → /auth/callback


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('google_token', None)
    return redirect(url_for('home'))

@app.route('/auth/callback') 
def authorized():
    token = google.authorize_access_token()
    if token is None:
        return 'Access denied: reason={} error={}'.format(
            request.args.get('error_reason'),
            request.args.get('error_description')
        ) # If token is missing (maybe user said "No" or an error happened),

    session['google_token'] = token
    user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
    session['user'] = user_info
    email = user_info.get('email')

    # Check if the email ends with "study.iitm.ac.in"
    if not email.endswith('study.iitm.ac.in'):
        flash('Access denied: unauthorized email domain. Please login again with a valid email address.')
        session.pop('user', None)  # Clear the user session
        session.pop('google_token', None)  # Clear the token session
        return redirect(url_for('home'))

    session['user'] = user_info
    return redirect(url_for('home'))

if __name__ == '__main__':
    vector_store_path = "vector_store"
    os.makedirs(vector_store_path, exist_ok=True)

    if len(os.listdir(vector_store_path))==0:
        # Step 1: Create a vector store
        student_handbook_url = "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub#h.104s7slbfmp7"
        grading_doc_url = "https://docs.google.com/document/d/e/2PACX-1vRBH1NuM3ML6MH5wfL2xPiPsiXV0waKlUUEj6C7LrHrARNUsAEA1sT2r7IHcFKi8hvQ45gSrREnFiTT/pub?urp=gmail_link"
        vector_store, grading_doc_chunks = create_vector_store_from_docs(student_handbook_url, grading_doc_url)
        vector_store.save_local(vector_store_path)
    else:
        vector_store = load_vector_store(vector_store_path)
        # Load grading_doc_chunks from file if needed
        import json
        with open("grading_doc_chunks.json", "r", encoding="utf-8") as f:
            grading_doc_chunks = json.load(f)

    app.run(debug=True)
    # app.run(host='0.0.0.0', port=80, debug=True)
