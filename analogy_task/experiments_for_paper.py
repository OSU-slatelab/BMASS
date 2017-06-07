import os
import numpy as np
import config
from BMASS import settings
from analogy_task.embedding_wrapper import EmbeddingWrapper
from analogy_task.analogy_model import Mode
from analogy_task.task import analogyTask
from lib import util, log, preprocessing, embeddings
from lib.prm import PersistentResultsMatrix as PRM

def saveResults(results_dir, relation, results):
    prms = [
        PRM(1, path=os.path.join(results_dir, '%s.acc.npy' % relation.replace(': ', '-'))),
        PRM(1, path=os.path.join(results_dir, '%s.map.npy' % relation.replace(': ', '-'))),
        PRM(1, path=os.path.join(results_dir, '%s.mrr.npy' % relation.replace(': ', '-'))),
        PRM(1, path=os.path.join(results_dir, '%s.cor.npy' % relation.replace(': ', '-'))),
        PRM(1, path=os.path.join(results_dir, '%s.ttl.npy' % relation.replace(': ', '-'))),
    ]

    (correct, MAP, MRR, total, skipped, _) = results
    if total == 0:
        for i in range(5):
            prms[i][0] = -1
    else:
        prms[0][0] = float(correct) / total
        prms[1][0] = MAP
        prms[2][0] = MRR
        prms[3][0] = correct
        prms[4][0] = total-skipped

    for prm in prms:
        prm.save()

def evaluate(embedf, analogy_file, setting, freqtermf, unigrams, analogy_method,
        log=log, predictions_file=None, predictions_file_mode='w',
        report_top_k=5, glove_vocab=None, clean_vocab=False):
    t_main = log.startTimer()

    # read main embeddings file
    t_sub = log.startTimer('Reading embeddings from %s...' % embedf, newline=False)
    if not glove_vocab:
        embeds = embeddings.read(embedf)
    else:
        embeds = embeddings.read(embedf, format=embeddings.Format.Glove, vocab=glove_vocab)

    if clean_vocab:
        clean = lambda k: ' '.join(preprocessing.tokenize(k))
        keys = set(embeds.keys())
        cleaned_embeds, keys_set_aside = {}, []
        # first, find keys that are already clean
        for key in keys:
            clean_key = clean(key)
            if clean_key == key: cleaned_embeds[key] = embeds[key]
            else: keys_set_aside.append(key)
        # put the keys set aside into deterministic order (preferring "To" to "TO")
        keys_set_aside.sort()
        keys_set_aside.reverse()
        # then, find ones not already in the embedding list and add them
        for key in keys_set_aside:
            clean_key = clean(key)
            if cleaned_embeds.get(clean_key, None) is None:
                cleaned_embeds[clean_key] = embeds[key]
        embeds = cleaned_embeds

    log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(embeds))
                    
    # unit-norm each embedding to simplify cosine similarity
    t_sub = log.startTimer('Norming embeddings...', newline=False)
    for (k,v) in embeds.items():
        embeds[k] = np.array(embeds[k]) / np.linalg.norm(embeds[k])
    log.stopTimer(t_sub, message='Complete ({0:.2f}s).')

    # finally, if using non-unigram data, construct the embedding vocabulary by averaging
    # the token embeddings for all known strings; treat word embeddings like backoff
    if not unigrams:
        t_sub = log.startTimer('Reading frequent term vocabulary...', newline=False)
        str_vocab = util.readList(freqtermf, encoding='utf-8')
        log.stopTimer(t_sub, message='Read %d vocabulary terms ({0:.2f}s).' % len(str_vocab))

        t_sub = log.startTimer('Constructing alias vocabulary...', newline=False)
        backoff_embeds = embeds
        averaged_embeds = {}
        for known_str in str_vocab:
            token_embeds = []
            for t in known_str.split():
                if not backoff_embeds.get(t, None) is None:
                    token_embeds.append(backoff_embeds[t])
            if len(token_embeds) > 0:
                averaged_embeds[known_str] = np.mean(token_embeds, axis=0)
        embeds = averaged_embeds
        log.stopTimer(t_sub, message='Complete ({0:.2f}s).')
    # if using unigram data, just take the word embeddings as the candidate vocabulary
    else: 
        backoff_embeds = None

    # abstract away the embedding access
    t_sub = log.startTimer('Building embedding wrapper...', newline=False)
    emb_wrapper = EmbeddingWrapper(embeds, backoff_embeds=backoff_embeds)
    log.stopTimer(t_sub, message='Complete ({0:.2f}s).')

    results = analogyTask(analogy_file, setting, emb_wrapper, log=log, predictions_file=predictions_file, predictions_file_mode=predictions_file_mode, report_top_k=report_top_k)

    log.stopTimer(t_main, message='Program complete in {0:.2f}s.')

    return results


if __name__ == '__main__':

    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog ANALOGY_FILE RESULTS_DIR',
                description='Run the analogy task on analogies in ANALOGY_FILE')
        parser.add_option('--frequent-term-list', dest='freqtermf',
                help='list of frequent terms to use as completion vocabulary',
                default=config.FREQUENT_TERMS)
        parser.add_option('--setting', dest='setting',
                help='BMASS variant',
                type='choice', choices=['All-Info', 'Multi-Answer', 'Single-Answer'])
        parser.add_option('-l', '--logfile', dest='logfile',
                help='logfile')
        parser.add_option('--predictions-file', dest='predictions_file',
                help='file to write predictions for individual analogies to')
        parser.add_option('--predictions-top-k', dest='report_top_k',
                help='number of predictions to log in the predictions file (default: %default)',
                type='int', default=5)
        parser.add_option('--unigrams', dest='unigrams',
                help='evaluate on unigram data (don\'t use MWE candidates)',
                action='store_true', default=False)
        parser.add_option('--unigram-mwe-compare', dest='unigram_mwe_comparison',
                help='evaluate on unigram data (using MWE candidates)',
                action='store_true', default=False)
        parser.add_option('--analogy-method', dest='analogy_method',
                help='method to use for analogy completion',
                type='int', default=Mode.ThreeCosAdd)
        (options, args) = parser.parse_args()
        if len(args) != 2 \
                or (not options.unigrams and not options.freqtermf):
            parser.print_help()
            exit()

        analogy_file, results_dir = args

        if options.setting == 'Single-Answer': options.setting = settings.SINGLE_ANSWER
        elif options.setting == 'Multi-Answer': options.setting = settings.MULTI_ANSWER
        elif options.setting == 'All-Info': options.setting = settings.ALL_INFO

        return (analogy_file, options.setting, results_dir,
            options.freqtermf, options.unigrams, options.unigram_mwe_comparison, 
            options.analogy_method, 
            options.logfile, options.predictions_file, options.report_top_k,
        )
    
    (analogy_file, setting, results_dir, freqtermf, unigrams, unigram_mwe_comparison, 
        analogy_method, logfile, predictions_file, report_top_k) = args = _cli()
    log.start(logfile=logfile, stdout_also=True)

    # if storing predictions, clear the file here
    if predictions_file:
        with open(predictions_file, 'w') as stream:
            pass

    for (embedf, label, vocab_is_dirty) in config.LABELED_EMBEDDINGS:
        if type(embedf) is tuple:
            (embedf, glove_vocabf) = embedf
        else:
            glove_vocabf = None

        these_results_dir = os.path.join(results_dir, os.path.splitext(os.path.basename(embedf))[0])
        if not os.path.isdir(these_results_dir):
            os.mkdir(these_results_dir)

        log.writeln(('\n\n\n{0}\nEmbeddings: %s\n{0}\n\n' % label).format('-'*79))

        if predictions_file:
            with open(predictions_file, 'a') as stream:
                stream.write(('\n\n\n{0}\nEmbeddings: %s\n{0}\n\n\n' % label).format('-'*79))

        results = evaluate(embedf, analogy_file, setting, freqtermf,
            unigrams, analogy_method, log=log, 
            predictions_file=predictions_file, predictions_file_mode='a', report_top_k=report_top_k,
            glove_vocab=glove_vocabf, clean_vocab=vocab_is_dirty)
        
        for (relation, rel_results) in results.items():
            saveResults(these_results_dir, relation, rel_results)
