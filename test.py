from src.kasre_detection import KasreEzafe


if __name__ == "__main__":
    
    ke_obj = KasreEzafe()
    sample_text = 'منابع خبری از محاصره منزل رهبر مخالفان ونزوئلا در پایتخت توسط نیروهای امنیتی خبر می‌دهند'
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
        if token.startswith("##"):
            if previous_label in ['B-WORD', 'I-WORD'] and previous_index == index - 1:
                current_word += token.replace("##", "")
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

