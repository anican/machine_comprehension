import argparse
import logging
import os
import shutil
import sys

from allennlp.data.dataset_readers import SquadReader
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data import Vocabulary
from allennlp.models.reading_comprehension.bidaf import (
    BidirectionalAttentionFlow)
from allennlp.training.metrics import (
    BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1)
from allennlp.nn.util import move_to_device
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.nn.functional import nll_loss
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from rnn_attention_rc.data import load_embeddings, read_data
from rnn_attention_rc.demo import run_demo
from rnn_attention_rc.models.attention_rnn import AttentionRNN
from rnn_attention_rc.models.cbow import CBOW
from rnn_attention_rc.models.rnn import RNN

logger = logging.getLogger(__name__)
get_best_span = BidirectionalAttentionFlow.get_best_span

# Dictionary of model type strings to model classes
MODEL_TYPES = {
    "attention": AttentionRNN,
    # For compatibility with serialization
    "attentionrnn": AttentionRNN,
    "cbow": CBOW,
    "rnn": RNN,
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--squad-train-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "train_small.json"),
                        help="Path to the SQuAD training data.")
    parser.add_argument("--squad-dev-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "val_small.json"),
                        help="Path to the SQuAD dev data.")
    parser.add_argument("--squad-test-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "test_small.json"),
                        help="Path to the SQuAD test data.")
    parser.add_argument("--glove-path", type=str,
                        default=os.path.join(project_root, "glove",
                                             "glove.6B.50d.txt"),
                        help="Path to word vectors in GloVe format.")
    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="cbow",
                        choices=["cbow", "rnn", "attention"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--max-passage-length", type=int, default=150,
                        help="Maximum number of words in the passage.")
    parser.add_argument("--max-question-length", type=int, default=15,
                        help="Maximum number of words in the question.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and Attention models.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate to use.")
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    parser.add_argument("--demo", action="store_true",
                        help="Run the interactive web demo.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to use for web demo.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to use for web demo.")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load a model from checkpoint and evaluate it on test data.
    if args.load_path:
        logger.info("Loading saved model from {}".format(args.load_path))

        # If evaluating with CPU, force all tensors to CPU.
        # This lets us load models trained on the GPU and evaluate with CPU.
        saved_state_dict = torch.load(args.load_path,
                                      map_location=None if args.cuda
                                      else lambda storage, loc: storage)

        # Extract the contents of the state dictionary.
        model_type = saved_state_dict["model_type"]
        model_weights = saved_state_dict["model_weights"]
        model_init_arguments = saved_state_dict["init_arguments"]
        model_global_step = saved_state_dict["global_step"]

        # Reconstruct a model of the proper type with the init arguments.
        saved_model = MODEL_TYPES[model_type.lower()](**model_init_arguments)
        # Load the weights
        saved_model.load_state_dict(model_weights)
        # Set the global step
        saved_model.global_step = model_global_step

        logger.info("Successfully loaded model!")

        # Move model to GPU if CUDA is on.
        if args.cuda:
            saved_model = saved_model.cuda()

        # Load the serialized train_vocab.
        vocab_dir = os.path.join(os.path.dirname(args.load_path),
                                 "train_vocab")
        logger.info("Loading train vocabulary from {}".format(vocab_dir))
        train_vocab = Vocabulary.from_files(vocab_dir)
        logger.info("Successfully loaded train vocabulary!")

        if args.demo:
            # Run the demo with the loaded model.
            run_demo(saved_model, train_vocab, args.host, args.port,
                     args.cuda)
            sys.exit(0)

        # Evaluate on the SQuAD test set.
        logger.info("Reading SQuAD test set at {}".format(
            args.squad_test_path))
        squad_reader = SquadReader()
        test_dataset = squad_reader.read(args.squad_test_path)
        logger.info("Read {} test examples".format(
            len(test_dataset)))
        # Filter out examples with passage length greater than
        # max_passage_length or question length greater than
        # max_question_length
        logger.info("Filtering out examples in test set with "
                    "passage length greater than {} or question "
                    "length greater than {}".format(
                        args.max_passage_length, args.max_question_length))
        test_dataset = [
            instance for instance in tqdm(test_dataset) if
            (len(instance.fields["passage"].tokens) <=
             args.max_passage_length) and
            (len(instance.fields["question"].tokens) <=
             args.max_question_length)]
        logger.info("{} test examples remain after filtering".format(
            len(test_dataset)))

        # Evaluate the model on the test set.
        logger.info("Evaluating model on the test set")
        (loss, span_start_accuracy, span_end_accuracy,
         span_accuracy, em, f1) = evaluate(
             saved_model, test_dataset, args.batch_size,
             train_vocab, args.cuda)
        # Log metrics to console.
        logger.info("Done evaluating on test set!")
        logger.info("Test Loss: {:.4f}".format(loss))
        logger.info("Test Span Start Accuracy: {:.4f}".format(
            span_start_accuracy))
        logger.info("Test Span End Accuracy: {:.4f}".format(span_end_accuracy))
        logger.info("Test Span Accuracy: {:.4f}".format(span_accuracy))
        logger.info("Test EM: {:.4f}".format(em))
        logger.info("Test F1: {:.4f}".format(f1))
        sys.exit(0)

    if not args.save_dir:
        raise ValueError("Must provide a value for --save-dir if training.")

    try:
        if os.path.exists(args.save_dir):
            # save directory already exists, do we really want to overwrite?
            input("Save directory {} already exists. Press <Enter> "
                  "to clear, overwrite and continue , or "
                  "<Ctrl-c> to abort.".format(args.save_dir))
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    # Write tensorboard logs to save_dir/logs.
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir)

    # Read the training and validaton dataset, and get a vocabulary
    # from the train set.
    train_dataset, train_vocab, validation_dataset = read_data(
        args.squad_train_path, args.squad_dev_path, args.max_passage_length,
        args.max_question_length, args.min_token_count)

    # Save the train_vocab to a file.
    vocab_dir = os.path.join(args.save_dir, "train_vocab")
    logger.info("Saving train vocabulary to {}".format(vocab_dir))
    train_vocab.save_to_files(vocab_dir)

    # Read GloVe embeddings.
    embedding_matrix = load_embeddings(args.glove_path, train_vocab)

    # Create model of the correct type.
    if args.model_type == "cbow":
        logger.info("Building CBOW model")
        model = CBOW(embedding_matrix)
    if args.model_type == "rnn":
        logger.info("Building RNN model")
        model = RNN(embedding_matrix, args.hidden_size, args.dropout)
    if args.model_type == "attention":
        logger.info("Building attention RNN model")
        model = AttentionRNN(embedding_matrix, args.hidden_size,args.dropout)
    logger.info(model)

    # Move model to GPU if running with CUDA.
    if args.cuda:
        model = model.cuda()
    # Create the optimizer, and only update parameters where requires_grad=True
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=args.lr)
    # Train for the specified number of epochs.
    for i in tqdm(range(args.num_epochs), unit="epoch"):
        train_epoch(model, train_dataset, validation_dataset, train_vocab,
                    args.batch_size, optimizer, args.log_period,
                    args.validation_period, args.save_dir, log_dir,
                    args.cuda)


def train_epoch(model, train_dataset, validation_dataset, vocab,
                batch_size, optimizer, log_period, validation_period,
                save_dir, log_dir, cuda):
    """
    Train the model for one epoch.
    """
    # Set model to train mode (turns on dropout and such).
    model.train()
    # Create objects for calculating metrics.
    span_start_accuracy_metric = CategoricalAccuracy()
    span_end_accuracy_metric = CategoricalAccuracy()
    span_accuracy_metric = BooleanAccuracy()
    squad_metrics = SquadEmAndF1()
    # Create Tensorboard logger.
    writer = SummaryWriter(log_dir)

    # Build iterater, and have it bucket batches by passage / question length.
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[("passage", "num_tokens"),
                                            ("question", "num_tokens")])
    # Index the instances with the vocabulary.
    # This converts string tokens to numerical indices.
    iterator.index_with(vocab)
    num_training_batches = iterator.get_num_batches(train_dataset)
    # Get a generator of train batches.
    train_generator = tqdm(iterator(train_dataset, num_epochs=1),
                           total=num_training_batches, leave=False)
    log_period_losses = 0

    for batch in train_generator:
        # move the data to cuda if available
        batch = move_to_device(batch, cuda_device=0 if cuda else -1)
        # Extract the relevant data from the batch.
        passage = batch["passage"]["tokens"]
        question = batch["question"]["tokens"]
        span_start = batch["span_start"]
        span_end = batch["span_end"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]
        softmax_start_logits = output_dict["softmax_start_logits"]
        softmax_end_logits = output_dict["softmax_end_logits"]

        # Calculate loss for start and end indices.
        loss = nll_loss(softmax_start_logits, span_start.view(-1))
        loss += nll_loss(softmax_end_logits, span_end.view(-1))
        log_period_losses += loss.item()

        # Backprop and take a gradient step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.global_step += 1

        # Calculate categorical span start and end accuracy.
        span_start_accuracy_metric(start_logits, span_start.view(-1))
        span_end_accuracy_metric(end_logits, span_end.view(-1))
        # Compute the best span, and calculate overall span accuracy.
        best_span = get_best_span(start_logits, end_logits)
        span_accuracy_metric(
            best_span, torch.cat([span_start, span_end], -1))
        # Calculate EM and F1 scores
        calculate_em_f1(best_span, metadata, passage.size(0),
                        squad_metrics)

        if model.global_step % log_period == 0:
            # Calculate metrics on train set.
            loss = log_period_losses / log_period
            span_start_accuracy = span_start_accuracy_metric.get_metric(
                reset=True)
            span_end_accuracy = span_end_accuracy_metric.get_metric(reset=True)
            span_accuracy = span_accuracy_metric.get_metric(reset=True)
            em, f1 = squad_metrics.get_metric(reset=True)
            tqdm_description = _make_tqdm_description(
                loss, em, f1)
            # Log training statistics to progress bar
            train_generator.set_description(tqdm_description)
            # Log training statistics to Tensorboard
            log_to_tensorboard(writer, model.global_step, "train",
                               loss, span_start_accuracy, span_end_accuracy,
                               span_accuracy, em, f1)
            log_period_losses = 0

        if model.global_step % validation_period == 0:
            # Calculate metrics on validation set.
            (loss, span_start_accuracy, span_end_accuracy,
             span_accuracy, em, f1) = evaluate(
                 model, validation_dataset, batch_size, vocab, cuda)
            # Save a checkpoint.
            save_name = ("{}_step_{}_loss_{:.3f}_"
                         "em_{:.3f}_f1_{:.3f}.pth".format(
                             model.__class__.__name__, model.global_step,
                             loss, em, f1))
            save_model(model, save_dir, save_name)
            # Log validation statistics to Tensorboard.
            log_to_tensorboard(writer, model.global_step, "validation",
                               loss, span_start_accuracy, span_end_accuracy,
                               span_accuracy, em, f1)


def evaluate(model, evaluation_dataset, batch_size, vocab, cuda):
    """
    Evaluate a model on an evaluation dataset.
    """
    # Set model to evaluation mode (turns off dropout and such)
    model.eval()
    # Create objects for calculating metrics.
    span_start_accuracy = CategoricalAccuracy()
    span_end_accuracy = CategoricalAccuracy()
    span_accuracy = BooleanAccuracy()
    squad_metrics = SquadEmAndF1()

    # Build iterater, and have it bucket batches by passage / question length.
    evaluation_iterator = BasicIterator(batch_size=batch_size)
    # Index the instances with the vocabulary.
    # This converts string tokens to numerical indices.
    evaluation_iterator.index_with(vocab)
    # Get a generator of train batches.
    num_evaluation_batches = evaluation_iterator.get_num_batches(
        evaluation_dataset)
    evaluation_generator = tqdm(
        evaluation_iterator(
            evaluation_dataset, num_epochs=1, shuffle=False),
            total=num_evaluation_batches, leave=False)
    batch_losses = 0
    for batch in evaluation_generator:
        # move the data to cuda if available
        batch = move_to_device(batch, cuda_device=0 if cuda else -1)
        # Extract the relevant data from the batch.
        passage = batch["passage"]["tokens"]
        question = batch["question"]["tokens"]
        span_start = batch["span_start"]
        span_end = batch["span_end"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]
        softmax_start_logits = output_dict["softmax_start_logits"]
        softmax_end_logits = output_dict["softmax_end_logits"]

        # Calculate loss for start and end indices.
        loss = nll_loss(softmax_start_logits, span_start.view(-1))
        loss += nll_loss(softmax_end_logits, span_end.view(-1))
        batch_losses += loss.item()

        # Calculate categorical span start and end accuracy.
        span_start_accuracy(start_logits, span_start.view(-1))
        span_end_accuracy(end_logits, span_end.view(-1))
        # Compute the best span, and calculate overall span accuracy.
        best_span = get_best_span(start_logits, end_logits)
        span_accuracy(best_span, torch.cat([span_start, span_end], -1))
        # Calculate EM and F1 scores
        calculate_em_f1(best_span, metadata, passage.size(0),
                        squad_metrics)

    # Set the model back to train mode.
    model.train()
    
    # loss = batch_losses / num_evaluation_batches
    # em, f1 = squad_metrics.get_metric(reset=True)
    # tqdm_description = _make_tqdm_description(
    #     loss, em, f1)
    # # Log training statistics to progress bar
    # # evaluation_generator.set_description(tqdm_description)

    # Extract the values from the metrics objects
    average_span_start_accuracy = span_start_accuracy.get_metric()
    average_span_end_accuracy = span_end_accuracy.get_metric()
    average_span_accuracy = span_accuracy.get_metric()
    average_em, average_f1 = squad_metrics.get_metric()
    return (batch_losses / num_evaluation_batches,
            average_span_start_accuracy,
            average_span_end_accuracy,
            average_span_accuracy,
            average_em,
            average_f1)


def calculate_em_f1(best_span, metadata, batch_size,
                    squad_metrics):
    """
    Calculates EM and F1 scores.
    """
    if metadata is not None:
        best_span_str = []
        for i in range(batch_size):
            passage_str = metadata[i]['original_passage']
            offsets = metadata[i]['token_offsets']
            predicted_span = tuple(best_span[i].data.cpu().numpy())
            start_offset = offsets[predicted_span[0]][0]
            end_offset = offsets[predicted_span[1]][1]
            best_span_string = passage_str[start_offset:end_offset]
            best_span_str.append(best_span_string)
            answer_texts = metadata[i].get('answer_texts', [])
            if answer_texts:
                squad_metrics(best_span_string, answer_texts)


def log_to_tensorboard(writer, step, prefix, loss, span_start_accuracy,
                       span_end_accuracy, span_accuracy,
                       em, f1):
    """
    Log metrics to Tensorboard.
    """
    writer.add_scalar("{}/loss".format(prefix), loss, step)
    writer.add_scalar("{}/span_start_accuracy".format(prefix),
                      span_start_accuracy, step)
    writer.add_scalar("{}/span_end_accuracy".format(prefix),
                      span_end_accuracy, step)
    writer.add_scalar("{}/span_accuracy".format(prefix),
                      span_accuracy, step)
    writer.add_scalar("{}/EM".format(prefix), em, step)
    writer.add_scalar("{}/F1".format(prefix), f1, step)


def save_model(model, save_dir, save_name):
    """
    Save a model to the disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_weights = model.state_dict()
    serialization_dictionary = {
        "model_type": model.__class__.__name__,
        "model_weights": model_weights,
        "init_arguments": model.init_arguments,
        "global_step": model.global_step
    }

    save_path = os.path.join(save_dir, save_name)
    torch.save(serialization_dictionary, save_path)


def _make_tqdm_description(average_loss, average_em, average_f1):
    """
    Build the string to use as the tqdm progress bar description.
    """
    metrics = {
        "Train Loss": average_loss,
        "Train EM": average_em,
        "Train F1": average_f1
    }
    return ", ".join(["%s: %.3f" % (name, value) for name, value
                      in metrics.items()]) + " ||"


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
