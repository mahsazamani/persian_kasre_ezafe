from flask import Flask, request, jsonify
from src.kasre_detection import KasreEzafe

app = Flask(__name__)

ke_obj = KasreEzafe()

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    
    # Check if 'text' is in the JSON body
    if 'text' not in data:
        return jsonify({'error': 'Text input is required'}), 400
    
    sample_text = data['text']
    
    output_wp, _ = ke_obj.predict(sample_text)

    words_list = []
    final_list = []

    # Collect tokens along with their labels and indexes
    for index, (token, label) in enumerate(output_wp):
        if label in ['B-WORD', 'I-WORD']:
            words_list.append((index, token, label))  # Store index, token, and label

    # Combine subwords into full words
    current_word = ""
    previous_index = None
    previous_label = None

    for index, token, label in words_list:
        if token.startswith("##"):  # This is a subword token
            # Combine only if the previous token was B-WORD or I-WORD
            if previous_label in ['B-WORD', 'I-WORD'] and previous_index == index - 1:
                current_word += token.replace("##", "")  # Combine with the previous part

        else:
            if current_word:
                final_list.append(current_word)
            current_word = token

        # Update previous values
        previous_index = index
        previous_label = label  # Update previous_label to the current token's label

    if current_word:
        final_list.append(current_word)

    print("Final Output:", final_list)
    
    
    return jsonify({'result': final_list})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)
