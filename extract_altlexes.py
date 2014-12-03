import pdtbsql as p

cb = p.CorpusBuilder()

c = cb.extract(relation='AltLex')


select distinct rawtext from annotations where relation = 'AltLex'
