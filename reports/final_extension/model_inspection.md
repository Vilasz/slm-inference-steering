# Inspeção dos modelos

qwen
- Revisão de pesos/tokenizer: 2e1fd397ee46e1388853d2af2c993145b0f1098a
- Arquitetura: ['Qwen2ForCausalLM']
- Camadas/hidden: 28/1536
- Tokenizer: Qwen2TokenizerFast
- BOS/EOS/PAD: (None, 151645, 151643)
- EOS generation_config: [151645, 151643]
- Vocabulário: 151665
- Fonte: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/tree/2e1fd397ee46e1388853d2af2c993145b0f1098a

deepseek
- Revisão de pesos/tokenizer: e063262dac8366fc1f28a4da0ff3c50ea66259ca
- Arquitetura: ['LlamaForCausalLM']
- Camadas/hidden: 24/2048
- Tokenizer: LlamaTokenizerFast
- BOS/EOS/PAD: (32013, 32021, 32014)
- EOS generation_config: 32021
- Vocabulário: 32022
- Fonte: https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct/tree/e063262dac8366fc1f28a4da0ff3c50ea66259ca

As diferenças de vocabulário e tokens impedem comparar entropias como se fossem a mesma distribuição. Os contrastes são incrementos H sobre B dentro de cada modelo.
Pesos BF16, projeções FP32, atenção eager; sem quantização; classes nativas Transformers sem execução de código remoto.
DeepSeek usa licença de pesos DeepSeek Model License; o código oficial usa MIT. Qwen usa Apache-2.0.
Fontes: https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct e https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.
O template DeepSeek inclui BOS e termina com newline após Response; a implementação preserva o texto e evita adicionar um segundo BOS.
A mensagem de sistema histórica substitui o sistema padrão DeepSeek, conforme suportado pelo template oficial. A semântica é compartilhada com Qwen.
