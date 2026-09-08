"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(corpus):
    # TODO: build a char-level vocab dict with 'stoi' and 'itos' fields, ids in sorted order from 0
    stoi = {}
    ind = 0
    for ch in sorted(corpus):
        if ch not in stoi:
            stoi[ch] = ind
            ind += 1

    itos = {ind:ch for ch, ind in stoi.items()}

    return dict(
        stoi=stoi,
        itos=itos
    )

# Step 2 - encode_string
def encode_string(text, vocab):
    # TODO: convert text into a list of integer token ids using vocab['stoi'].
    return [vocab['stoi'][ch] for ch in text]

# Step 3 - decode_ids
def decode_ids(ids, vocab):
    # TODO: decode a sequence of token ids back into the original string using vocab['itos'].
    return "".join([vocab['itos'][ind] for ind in ids])

# Step 4 - embed_tokens
import torch

def embed_tokens(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids.

    Args:
        token_ids: LongTensor of shape (T,).
        token_embedding: FloatTensor of shape (vocab_size, d_model).

    Returns:
        FloatTensor of shape (T, d_model).
    """
    # TODO: select the embedding row for each token id and return (T, d_model)
    return token_embedding[token_ids]

# Step 5 - add_positional_embeddings
import torch

def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    # TODO: add the appropriate slice of pos_embedding to token_embeds and return the sum.
    T, _ = token_embeds.shape
    return token_embeds + pos_embedding[start_pos:start_pos+T]

# Step 6 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    # TODO: compute x @ weight and add bias if provided
    out = x @ weight
    if bias is not None:
        out += bias

    return out

# Step 7 - compute_attention_scores
def compute_attention_scores(queries, keys):
    # TODO: return the (Tq, Tk) matrix of raw dot-product scores between queries and keys.
    return queries @ keys.T

# Step 8 - scale_attention_scores
def scale_attention_scores(scores, d_head):
    # TODO: scale raw attention scores by 1/sqrt(d_head) for numerical stability.
    scale = 1/(d_head**0.5)
    return scores * scale

# Step 9 - apply_causal_mask
def apply_causal_mask(scores, query_offset=0):
    # TODO: mask entries where key index > query_offset + query row index with -inf.
    mask = np.full((scores.shape[-2:]), -np.inf)
    mask = np.triu(mask, k=query_offset+1)

    return scores + mask

# Step 10 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    # TODO: apply a numerically stable softmax along the last axis of masked_scores
    shifted = np.exp(masked_scores - masked_scores.max(axis=-1, keepdims=True))
    return shifted/shifted.sum(axis=-1, keepdims=True)

# Step 11 - weighted_value_sum
import numpy as np

def weighted_value_sum(attn_weights, values):
    # TODO: combine attention weights (Tq, Tk) with values (Tk, d_head) into context (Tq, d_head).
    return attn_weights @ values

# Step 12 - project_qkv
import numpy as np

def project_qkv(x, attn_params):
    # TODO: project x into query, key, value tensors using attn_params
    q = linear_projection(x, attn_params['Wq'], attn_params.get('bq', None))
    k = linear_projection(x, attn_params['Wk'], attn_params.get('bk', None))
    v = linear_projection(x, attn_params['Wv'], attn_params.get('bv', None))

    return q, k, v

# Step 13 - append_kv_cache
def append_kv_cache(kv_cache, new_k, new_v):
    # TODO: extend the per-layer KV cache by appending new_k and new_v along the time axis.
    if kv_cache['k'] is None:
        kv_cache['k'] = new_k
    else:
        kv_cache['k'] = np.vstack((kv_cache['k'], new_k))

    if kv_cache['v'] is None:
        kv_cache['v'] = new_v
    else:
        kv_cache['v'] = np.vstack((kv_cache['v'], new_v))

    return kv_cache

# Step 14 - scaled_dot_product_attention_with_cache
import numpy as np

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    # TODO: combine score, scale, mask, softmax, and weighted value sum primitives.
    scores = compute_attention_scores(queries, kv_cache['k'])
    scores = scale_attention_scores(scores, queries.shape[-1])
    scores = apply_causal_mask(scores, query_offset)
    probs = softmax_attention_weights(scores)
    return weighted_value_sum(probs, kv_cache['v'])

# Step 15 - apply_output_projection
def apply_output_projection(context, attn_params):
    # TODO: project the attention context back to model dimension using attn_params['Wo'] and attn_params['bo'].
    return linear_projection(context, attn_params['Wo'], attn_params.get('bo', None))

# Step 16 - single_head_causal_self_attention
import numpy as np

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    # TODO: project to q,k,v; update kv_cache; run causal attention; output projection.
    q, k, v = project_qkv(x, attn_params)
    kv_cache = append_kv_cache(kv_cache, k, v)

    context = scaled_dot_product_attention_with_cache(q, kv_cache, query_offset)
    return apply_output_projection(context, attn_params), kv_cache

# Step 17 - ffn_first_layer_gelu
def ffn_first_layer_gelu(x, ffn_params):
    # TODO: apply the first linear layer of the FFN and then a GELU activation.
    z = linear_projection(x, ffn_params['W1'], ffn_params.get('b1', None))
    return z/2 * (1 + np.tanh(((2/np.pi)**0.5) * (z + 0.044715 * z**3)))

# Step 18 - ffn_second_layer
def ffn_second_layer(h, ffn_params):
    # TODO: apply the second FFN linear layer mapping (T, d_ff) back to (T, d_model).
    return linear_projection(h, ffn_params['W2'], ffn_params.get('b2', None))

# Step 19 - position_wise_feed_forward
def position_wise_feed_forward(x, ffn_params):
    # TODO: apply the FFN as two linear layers with a GELU between them
    h = ffn_first_layer_gelu(x, ffn_params)
    return ffn_second_layer(h, ffn_params)

# Step 20 - compute_mean_variance
import numpy as np

def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x."""
    # TODO: return (mean, var) reduced along the last axis with that axis kept
    return x.mean(axis=-1, keepdims=True), x.var(axis=-1, keepdims=True)

# Step 21 - layer_norm_apply
import numpy as np

def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    # TODO: normalize x across its last axis and apply the affine transform.
    mean, var = compute_mean_variance(x, eps)
    norm = (x - mean)/np.sqrt(var + eps)
    return ln_params['gamma'] * norm + ln_params['beta']

# Step 22 - residual_add_and_norm
import numpy as np

def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    # TODO: combine the residual connection with layer normalization over the feature axis.
    return layer_norm_apply(x + sublayer_output, ln_params, eps)

# Step 23 - transformer_block
def transformer_block(x, block_params, kv_cache, query_offset=0):
    # TODO: run attention (with KV cache) + FFN, each wrapped by residual add-and-norm
    if 'W_q' in block_params['attn']:
        block_params['attn']['Wq'] = block_params['attn']['W_q']
    if 'b_q' in block_params['attn']:
        block_params['attn']['bq'] = block_params['attn']['b_q']
    if 'W_k' in block_params['attn']:
        block_params['attn']['Wk'] = block_params['attn']['W_k']
    if 'b_k' in block_params['attn']:
        block_params['attn']['bk'] = block_params['attn']['b_k']
    if 'W_v' in block_params['attn']:
        block_params['attn']['Wv'] = block_params['attn']['W_v']
    if 'b_v' in block_params['attn']:
        block_params['attn']['bv'] = block_params['attn']['b_v']
    if 'W_o' in block_params['attn']:
        block_params['attn']['Wo'] = block_params['attn']['W_o']
    if 'b_o' in block_params['attn']:
        block_params['attn']['bo'] = block_params['attn']['b_o']

    # if block_params['attn']:
    #     try:
    attn, kv_cache = single_head_causal_self_attention(x, block_params['attn'], kv_cache, query_offset)
    x = residual_add_and_norm(x, attn, block_params['ln1'])
        # except KeyError:
        #     return x, kv_cache

    # if block_params['ffn']:
    #     try:
    out = position_wise_feed_forward(x, block_params['ffn'])
    x = residual_add_and_norm(x, out, block_params['ln2'])
        # except KeyError:
        #     return x, kv_cache
    return x, kv_cache

# Step 24 - lm_head_logits
def lm_head_logits(hidden, lm_head_params):
    # TODO: project hidden states to vocabulary logits via the LM head affine layer.
    return linear_projection(hidden, lm_head_params['W'], lm_head_params.get('b', None))

# Step 25 - greedy_next_token
def greedy_next_token(logits):
    # TODO: select the next token id by taking the argmax of the last logits row
    if len(logits.shape) > 1:
        logits = logits[-1]
    return int(np.argmax(logits))

# Step 26 - run_prefill
def run_prefill(prompt_ids, model_params):
    """Run prefill over the prompt tokens and build the initial KV cache per layer."""
    # TODO: embed tokens, add positional embeddings, run every block, apply final layer norm
    embeddings = embed_tokens(prompt_ids, model_params['token_embedding'])
    x = add_positional_embeddings(embeddings, model_params['pos_embedding'])
    kv_caches = []

    for block in model_params['blocks']:
        kv_cache = {'k': None, 'v': None}
        x, kv_cache = transformer_block(x, block, kv_cache)
        kv_caches.append(kv_cache)

    x = layer_norm_apply(x, model_params['ln_f'])
    logits = lm_head_logits(x, model_params['lm_head'])

    return dict(
        hidden=x,
        kv_caches=kv_caches,
        next_pos=len(prompt_ids)
    )

# Step 27 - decode_step
def decode_step(prev_token_id, kv_caches, next_pos, model_params):
    # TODO: run one autoregressive decode step and return next_token, logits, kv_caches, next_pos.
    token_embeds = embed_tokens([prev_token_id], model_params['token_embedding'])
    x = add_positional_embeddings(token_embeds, model_params['pos_embedding'], start_pos=next_pos)

    for i in range(len(kv_caches)):
        x, kv_caches[i] = transformer_block(x, model_params['blocks'][i], kv_caches[i], next_pos)

    x = layer_norm_apply(x, model_params['ln_f'])
    logits = lm_head_logits(x, model_params['lm_head'])

    next_token = greedy_next_token(logits)

    return dict(
        next_token=next_token,
        logits=logits[0],
        kv_caches=kv_caches,
        next_pos=next_pos+1
    )

# Step 28 - generate_with_state_log
def generate_with_state_log(prompt_ids, model_params, num_new_tokens):
    """Run prefill, then decode num_new_tokens tokens, logging each step's state."""
    # TODO: run prefill, then autoregressively decode while recording each step
    prefill = run_prefill(prompt_ids, model_params)
    kv_caches = prefill['kv_caches']
    next_pos = prefill['next_pos']

    x = prefill['hidden']
    logits = lm_head_logits(x[-1], model_params['lm_head'])
    prev_token_id = greedy_next_token(logits)
    step_state = dict(
        next_token=prev_token_id,
        logits=logits,
        kv_caches=kv_caches,
        next_pos=next_pos
    )

    if num_new_tokens == 0:
        return dict(
            generated_tokens=[],
            step_states=[]
        )

    step_states = [step_state]
    generated_tokens = [prev_token_id]

    for _ in range(num_new_tokens-1):
        decode = decode_step(prev_token_id, kv_caches, next_pos, model_params)
        step_states.append(decode)
        prev_token_id = decode['next_token']
        generated_tokens.append(prev_token_id)

        kv_caches = decode['kv_caches']
        next_pos = decode['next_pos']

    return dict(
        generated_tokens=generated_tokens,
        step_states=step_states
    )

# Step 29 - hash_tensor
import hashlib
import numpy as np

def hash_tensor(tensor):
    """Return a 32-byte SHA-256 digest of the tensor's shape, dtype, and contents."""
    # TODO: canonically serialize the tensor and return sha256(...).digest()
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(tensor).tobytes())
    hasher.update(str(tensor.shape).encode('utf-8'))
    return hasher.digest()

# Step 30 - commit_decode_step
def commit_decode_step(step_state):
    # TODO: build a 32-byte Merkle leaf digest committing to every field of one decode step.
    hasher = hashlib.sha256()
    hasher.update(hash_tensor(np.array(step_state['step_index'])))
    hasher.update(hash_tensor(np.array(step_state['input_token'])))
    hasher.update(hash_tensor(np.array(step_state['next_token'])))
    hasher.update(hash_tensor(step_state['logits']))
    hasher.update(hash_tensor(np.array(step_state['next_pos'])))

    for kv_cache in step_state['kv_caches']:
        hasher.update(hash_tensor(kv_cache['k']))
        hasher.update(hash_tensor(kv_cache['v']))

    return hasher.digest()

# Step 31 - hash_pair
import hashlib

def hash_pair(left_digest, right_digest):
    """Hash two child digests into a single parent digest."""
    # TODO: combine two child digests into a single parent digest via SHA-256
    hasher = hashlib.sha256()
    hasher.update(left_digest)
    hasher.update(right_digest)

    return hasher.digest()

# Step 32 - build_merkle_level
def build_merkle_level(nodes):
    # TODO: hash adjacent pairs of nodes, duplicating the last if odd, to form the next Merkle level.
    leaves = nodes
    if len(nodes)%2:
        leaves = nodes + [nodes[-1]]

    parents = []
    for i in range(0, len(leaves), 2):
        parents.append(hash_pair(leaves[i], leaves[i+1]))

    return parents

# Step 33 - build_merkle_tree
def build_merkle_tree(leaves):
    # TODO: build the full Merkle tree as a list of levels from the given leaf digests.
    levels = [leaves]
    while len(leaves) != 1:
        leaves = build_merkle_level(leaves)
        levels.append(leaves)

    return levels

# Step 34 - merkle_root
def merkle_root(tree):
    # TODO: return the Merkle root digest from a built tree (list of levels).
    return tree[-1][0]

# Step 35 - merkle_inclusion_proof
def merkle_inclusion_proof(tree, leaf_index):
    # TODO: walk from the leaf level upward, recording each sibling and its side
    siblings = []
    if len(tree) <= 1:
        return siblings

    for level in tree[:-1]:
        if leaf_index % 2 == 0:
            if leaf_index == len(level)-1:
                siblings.append({'sibling':level[leaf_index], 'is_right': True})
            else:
                siblings.append({'sibling':level[leaf_index+1], 'is_right': True})
        else:
            siblings.append({'sibling':level[leaf_index-1], 'is_right': False})

        leaf_index = leaf_index//2

    return siblings

# Step 36 - verify_merkle_inclusion_proof
def verify_merkle_inclusion_proof(leaf, leaf_index, proof, root):
    # TODO: walk from leaf to root using sibling digests and return True iff root matches.
    current = leaf
    for sibling in proof:
        if sibling['side'] == 'right':
            current = hash_pair(current, sibling['sibling'])
        else:
            current = hash_pair(sibling['sibling'], current)

    return current == root

# Step 37 - run_prover
def run_prover(model_params, prompt_ids, num_steps):
    # TODO: Generate num_steps tokens greedily and produce a Merkle leaf for every decode step.
    if num_steps == 0:
        return dict(
            output_tokens=[],
            step_states=[],
            leaves=[]
        )

    completion = generate_with_state_log(prompt_ids, model_params, num_steps)
    leaves = [commit_decode_step(step_state) for step_state in completion['step_states']]

    return dict(
        output_tokens=completion['generated_tokens'],
        step_states=completion['step_states'],
        leaves=leaves
    )

# Step 38 - assemble_public_transcript (not yet solved)
# TODO: implement

# Step 39 - sample_audit_positions (not yet solved)
# TODO: implement

# Step 40 - reexecute_audited_step (not yet solved)
# TODO: implement

# Step 41 - recompute_step_commitment (not yet solved)
# TODO: implement

# Step 42 - check_commitment_against_proof (not yet solved)
# TODO: implement

# Step 43 - check_token_matches_claim (not yet solved)
# TODO: implement

# Step 44 - run_spot_check_verification (not yet solved)
# TODO: implement

# Step 45 - tamper_transcript_flip_token (not yet solved)
# TODO: implement

# Step 46 - detection_probability (not yet solved)
# TODO: implement

# Step 47 - verifier_cost_fraction (not yet solved)
# TODO: implement

# Step 48 - show_tampered_transcript_rejected (not yet solved)
# TODO: implement

# Step 49 - sample_verifier_committee (not yet solved)
# TODO: implement

# Step 50 - collect_verifier_votes (not yet solved)
# TODO: implement

# Step 51 - aggregate_votes_majority (not yet solved)
# TODO: implement

# Step 52 - reward_honest_participants (not yet solved)
# TODO: implement

# Step 53 - slash_worker (not yet solved)
# TODO: implement

# Step 54 - assign_dual_role (not yet solved)
# TODO: implement

# Step 55 - run_honest_round (not yet solved)
# TODO: implement

# Step 56 - run_malicious_round (not yet solved)
# TODO: implement

# Step 57 - report_end_to_end_verification_cost (not yet solved)
# TODO: implement

