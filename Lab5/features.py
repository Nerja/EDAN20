import transition

__author__ = "Marcus Rodan"

def extract_set1_features(stack, queue, graph, feature_names, sentence):
    can_la      = transition.can_leftarc(stack, graph)
    can_re      = transition.can_reduce(stack, graph)
    stack_0_w   = stack[0]['form'] if stack else 'NULL'
    stack_0_pos = stack[0]['postag'] if stack else 'NULL'
    queue_0_w   = queue[0]['form'] if queue else 'NULL'
    queue_0_pos = queue[0]['postag'] if queue else 'NULL'
    return [can_la, can_re, stack_0_w, stack_0_pos, queue_0_w, queue_0_pos]

def extract_set2_features(stack, queue, graph, feature_names, sentence):
    stack_1_w   = stack[1]['form'] if len(stack) >= 2 else 'NULL'
    stack_1_pos = stack[1]['postag'] if len(stack) >= 2 else 'NULL'
    queue_1_w   = queue[1]['form'] if len(queue) >= 2 else 'NULL'
    queue_1_pos = queue[1]['postag'] if len(queue) >= 2 else 'NULL'
    return [stack_1_w, stack_1_pos, queue_1_w, queue_1_pos]

def find_stack_0_fw(stack, sentence):
    nill_dict = dict([('form','NULL'),('postag', 'NULL')])
    if(len(stack) < 1):
        return nill_dict
    else:
        stack_0_id = int(stack[0]['id']) #say id=15 then we need id=16
        if(len(sentence) < stack_0_id + 2):
            return nill_dict
        else:
            return sentence[stack_0_id + 1]

def extract_set3_features(stack, queue, graph, feature_names, sentence):
    stack_0_fw = find_stack_0_fw(stack, sentence)
    return [stack_0_fw['form'], stack_0_fw['postag']]

def extract(stack, queue, graph, feature_names, sentence):
    set1_values = extract_set1_features(stack, queue, graph, feature_names, sentence)
    set2_values = extract_set2_features(stack, queue, graph, feature_names, sentence)
    set3_values = extract_set3_features(stack, queue, graph, feature_names, sentence)
    values      = set1_values + set2_values + set3_values
    return dict(zip(feature_names, values))
