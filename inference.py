from typing import List

from pico_llm import *

def main():
    embed_size = 1024
    num_inner_layers = 1

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    # kgram_model = KGramMLPSeqModel(
    #     vocab_size=vocab_size,
    #     k=3,
    #     embed_size=embed_size,
    #     num_inner_layers=num_inner_layers,
    #     chunk_size=32
    # ).to(device)
    #
    # lstm_model = LSTMSeqModel(
    #     vocab_size=vocab_size,
    #     embed_size=embed_size,
    #     hidden_size=embed_size
    # ).to(device)
    # #
    # kv_transformer = TransformerModel(
    #     device=device
    # ).to(device)
    lstm_model = torch.load('weights/lstm_model.pth', weights_only=False).to(device)
    kgram_model = torch.load('weights/kgram_model.pth', weights_only=False).to(device)
    models = {
        'lstm': lstm_model,
        'kgram': kgram_model,
        # 'transformer': kv_transformer
    }
    sampling_modes = {
        'greedy': lambda model, prompt, tokens: generate_text(
                    model, enc, prompt, max_new_tokens=tokens, device=device,
                    top_p=None,
                ),
        'nucleus': lambda model, prompt, tokens: generate_text(
                    model, enc, prompt, max_new_tokens=tokens, device=device,
                    top_p=0.95,
                ),
        'normal': lambda model, prompt, tokens: generate_text(
                    model, enc, prompt, max_new_tokens=tokens, device=device,
                    top_p=1.0,
                )
    }
    print('> Enter lstm, kgram, or transformer to specify model.')
    model_name = 'Enter model'
    sample_type = 'normal'
    command = 'help'

    def handle_command(parts: List[str]):
        nonlocal model_name, models, sampling_modes, sample_type
        keyword = parts[0] if len(parts) > 0 else None
        parameter = parts[1] if len(parts) == 2 else None
        if keyword is None:
            return
        elif keyword == 'exit':
            return True
        elif keyword == 'help':
            print('> Commands:\n'
                  '  - exit\n'
                  '  - help\n'
                  '  - generate <token_count>\n'
                  '  - sample <greedy|nucleus|normal>')
        elif keyword in models.keys():
            model_name = keyword
        elif keyword == 'generate':
            if model_name not in models:
                print('> Select a model first.')
                return
            num_tokens = 20
            if parameter is not None:
                try:
                    num_tokens = int(parameter)
                except ValueError:
                    pass
            prompt = input('> Enter prompt (default=Once upon a): ')
            if prompt == '':
                prompt = 'Once upon a'
            print(f'> Generating {num_tokens} tokens:')
            with torch.no_grad():
                text, annotated = sampling_modes[sample_type](models[model_name], prompt, num_tokens)
            words = text.replace('\n', '').split(' ')
            i = 0
            line = ''
            while i < len(words):
                line += words[i] + ' '
                if len(line) > 120:
                    print('>', line)
                    line = ''
                i += 1
            print('>', line)
        elif keyword == 'sample':
            if parameter in sampling_modes:
                sample_type = parameter
            else:
                print('> Invalid sampling type.')

    while True:
        if handle_command(command.split(' ')):
            break
        command = input(f'({model_name}) >')


if __name__ == '__main__':
    main()
