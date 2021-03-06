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
        input_text = "Input provided: {}".format(text)
        doc = nlp(text)
        output = "Named Entities:\n"
        for ent in doc.ents:
            output += "{} {} {} {}\n".format(ent.label_, ent.text, ent.start_char, ent.end_char)

    return render_template('index.html', text='{}'.format(input_text), named_entities='{}'.format(output.replace('\n', '<br>')))


if __name__ == "__main__":
    app.run(debug=True)
