import logging

from allennlp.data.dataset_readers import SquadReader
from allennlp.data.dataset import Batch
from allennlp.models.reading_comprehension.bidaf import (
    BidirectionalAttentionFlow)
from allennlp.nn.util import move_to_device
from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)
get_best_span = BidirectionalAttentionFlow.get_best_span


def run_demo(model, train_vocab, host, port, cuda):
    """
    Run the web demo application.
    """
    app = Flask(__name__)
    squad_reader = SquadReader()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/_get_answer")
    def get_answer():
        # Take user input and convert to Instance
        user_context = request.args.get("context", "", type=str)
        user_question = request.args.get("question", "", type=str)
        input_instance = squad_reader.text_to_instance(
            question_text=user_question,
            passage_text=user_context)
        # Make a dataset from the instance
        dataset = Batch([input_instance])
        dataset.index_instances(train_vocab)
        batch = dataset.as_tensor_dict()
        batch = move_to_device(batch, cuda_device=0 if cuda else -1)
        # Extract relevant data from batch.
        passage = batch["passage"]["tokens"]
        question = batch["question"]["tokens"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]

        # Compute the best span
        best_span = get_best_span(start_logits, end_logits)

        # Get the string corresponding to the best span
        passage_str = metadata[0]['original_passage']
        offsets = metadata[0]['token_offsets']
        predicted_span = tuple(best_span[0].data.cpu().numpy())
        start_offset = offsets[predicted_span[0]][0]
        end_offset = offsets[predicted_span[1]][1]
        best_span_string = passage_str[start_offset:end_offset]

        # Return the best string back to the GUI
        return jsonify(answer=best_span_string)

    logger.info("Launching Demo...")
    app.run(port=port, host=host)
