from flask import Flask, render_template, request

from utils import dialog_gpt, save_to_pickle, load_from_pickle, load_keywords_from_csv, find_custom_answer

from config import (TMP_FILENAME_FOR_DIALOGUE_HELPER_DATA,
                    RESTART_KEYWORDS_PATH,
                    EXIT_KEYWORDS_PATH,
                    COS_SIM_THRESHOLD)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("maintemplate.html")


restart_dialogue = True
chat_history_ids = None
helper_data = {"restart_dialogue": restart_dialogue, "chat_history_ids": chat_history_ids}
save_to_pickle(helper_data, TMP_FILENAME_FOR_DIALOGUE_HELPER_DATA)
restart_keywords = load_keywords_from_csv(RESTART_KEYWORDS_PATH)
exit_keywords = load_keywords_from_csv(EXIT_KEYWORDS_PATH)


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')

    custom_answer = find_custom_answer(user_text, threshold=COS_SIM_THRESHOLD)
    if custom_answer:
        reply_text = custom_answer
    else:
        helper_data = load_from_pickle(TMP_FILENAME_FOR_DIALOGUE_HELPER_DATA)
        restart_dialogue = helper_data["restart_dialogue"]
        chat_history_ids = helper_data["chat_history_ids"]

        if any(w == user_text.strip().lower() for w in restart_keywords):
            reply_text = "Ok, let's start from scratch, I am ready"
            restart_dialogue = True
        elif any(w == user_text.strip().lower() for w in exit_keywords):
            reply_text = "Ok, bye! Just waiting if you type something..."
            restart_dialogue = True
        else:
            reply_text, chat_history_ids = dialog_gpt(user_text, chat_history_ids, restart_dialogue)
            restart_dialogue = False

        helper_data = {"restart_dialogue": restart_dialogue, "chat_history_ids": chat_history_ids}
        save_to_pickle(helper_data, TMP_FILENAME_FOR_DIALOGUE_HELPER_DATA)

    return reply_text


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
