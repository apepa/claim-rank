import numpy as np

def get_claim_scores(sentences):
	return np.array([np.array([1 if s.label>0 else 0] +
		[int(l) for l in s.labels]) for s in sentences])


def get_features(debate_sentences, all_indices, pipeline_func):
    all_sentences = [np.concatenate(debate_sentences[indices]) for indices in all_indices]
    
    pipeline = pipeline_func(all_sentences[0])
    
    Xs = [pipeline.fit_transform(all_sentences[0])] + [pipeline.transform(sentences) for sentences in all_sentences[1:]]
    
    return Xs

def get_targets(debate_results, all_indices):
	return [np.vstack(debate_results[indices]) for indices in all_indices]

def save_data(debate_sentences, crossvalidation_indices, getPipeline, name='crossval'):
	debate_results = np.array([get_claim_scores(debate_sentence) for debate_sentence in debate_sentences])
	folds = []
	for i, indices in enumerate(crossvalidation_indices):
	    Xs = get_features(debate_sentences, indices, getPipeline)
	    ys = get_targets(debate_results, indices)
	    folds.append(Xs + ys)
	np.save('folds/' + name, np.array(folds))

def load_data(name='crossval'):
	folds = np.load('folds/' + name + '.npy')
	return folds;
