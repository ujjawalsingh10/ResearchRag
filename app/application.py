from flask import Flask, render_template, url_for, redirect, request, session
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.environ.get('HF_TOKEN')

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace('\n', '<br>\n'))

app.jinja_env.filters['nl2br'] = nl2br

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'messages' not in session:
        session['messages'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('prompt')

        if user_input:
            messages = session['messages']
            messages.append({'role' : 'user', 'content': user_input})
            session['messages'] = messages

            try:
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({'query' : user_input})
                # response = qa_chain.run(user_input)
                result = response.get('result', "No response")

                messages.append({'role': 'assistant', 'content' : result})
                session['messages'] = messages
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template('index.html', messages=session['messages'], error = error_msg)
        
        return redirect(url_for('index'))

    return render_template('index.html', messages=session.get('messages', []))

@app.route('/clear')
def clear():
    session.pop('messages', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader = False)


