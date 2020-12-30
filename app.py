from flask import Flask, request, render_template
import spacy

app = Flask(__name__)

nlp = spacy.load('my_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ner',methods=['POST'])
def ner():
    input = [x for x in request.form.values()]
    for text in input:
        output += "{}\n\n".format(text)
        doc = nlp(text)
        for ent in doc.ents:
            output += "{} {} {} {}\n".format(ent.label_, ent.text, ent.start_char, ent.end_char)

    return render_template('index.html', named_entities='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
