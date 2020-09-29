import spacy

from spacy.tokens import Doc, Span, Token
from sentence_transformers import SentenceTransformer, models

class SentenceBert(object):

    @staticmethod
    def install_extensions():
        '''Creates the extensions on docs, spans, tokens'''
        def vectorise(sent):
            return sent.doc._.sentence_bert_model.encode([sent.text])[0]
        
        # create an extension where the model will be used
        Doc.set_extension('sentence_bert_model', default=None, force=True)

        # set the extension both on doc and span level. This will contain the computed vector
        Token.set_extension('sentence_bert', getter=vectorise, force=True)
        Span.set_extension('sentence_bert', getter=vectorise, force=True)
        Doc.set_extension('sentence_bert', getter=vectorise, force=True)

    
    @staticmethod
    def overwrite_vectors(doc):
        '''Pipeline component that overwrites the vectors'''
        doc.user_hooks["vector"] = lambda a: a._.sentence_bert
        doc.user_span_hooks["vector"] = lambda a: a._.sentence_bert
        doc.user_token_hooks["vector"] = lambda a: a._.sentence_bert
        return doc

    @staticmethod
    def get_sentence_transformer(name):
        try:
            model = SentenceTransformer(name)
        except:
            transformer_model = models.Transformer(name)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=True,
                                           pooling_mode_max_tokens=False)

            model = SentenceTransformer(modules=[transformer_model, pooling_model])

        return model

    @staticmethod
    def create_nlp(config, nlp=None):
        model = SentenceBert.get_sentence_transformer(config['name'])

        def add_model_to_doc(doc):
            doc._.sentence_bert_model = model
            return doc
        
        if not nlp:
            nlp = spacy.blank(config['spacy_base_model'])
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp.add_pipe(add_model_to_doc, name='sentencebert_add_model_to_doc', first=True)
        nlp.add_pipe(SentenceBert.overwrite_vectors, name='sentencebert_overwrite_vectors', after='sentencebert_add_model_to_doc')

        return nlp
