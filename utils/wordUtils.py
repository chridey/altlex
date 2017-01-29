import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn

r = re.compile("[\x80-\xff]")
def replaceNonAscii(s):
    return r.sub('XX', s)

lemmatizer = WordNetLemmatizer()
snowballStemmer = SnowballStemmer('english')

def lemmatize(words, poses):
    lemmas = []
    for index,word in enumerate(words):
        if poses[index] is None:
            lemmas.append(word)
            continue
        if poses[index].startswith('RB'):
            pos = wn.ADV
        elif poses[index].startswith('JJ'):
            pos = wn.ADJ
        elif poses[index].startswith('NN'):
            pos = wn.NOUN
        elif poses[index].startswith('VB'):            
            pos = wn.VERB
        else:
            lemmas.append(word)
            continue
        lemmas.append(lemmatizer.lemmatize(word, pos=pos))
    return lemmas

def findPhrase(phrase, source):
    try:
        start = [tuple(source[i:i+len(phrase)]) for i in range(len(source))].index(tuple(phrase))
    except ValueError:
        return None
    return start


'''
sqlite> select c1,count1,count2,count3 from (select c1,count1,count2 from (select connhead as c1,count(*) as count1 from annotations group by connhead) as s1 left outer join (select connhead as c2,count(*) as count2 from annotations where firstsemfirst like 'Contingency.Cause.Reason%' or secondsemfirst like 'Contingency.Cause.Reason%' group by connhead) as s2 on c1=c2) as s3 left outer join (select connhead as c3,count(*) as count3 from annotations where firstsemfirst like 'Contingency.Cause.Result%' or secondsemfirst  like 'Contingency.Cause.Result%' group by connhead) as s4 on c1=c3;
|22141|2564|1871
And|1||
accordingly|5||5
additionally|7||
after|577|50|
afterward|11||
also|1746||
alternatively|6||
although|328||
and|2999||177
as|743|334|2
as a result|78||78
as an alternative|2||
as if|16||
as long as|24||
as soon as|20||
as though|5||
as well|6||
because|858|854|
before|326||
before and after|1||
besides|19||
but|3308|1|1
by comparison|11||
by contrast|27||
by then|7|1|
consequently|10||10
conversely|2||
earlier|15||
either or|4||
else|1||
except|10||
finally|32||2
for|3|3|
for example|196||
for instance|98||
further|9||
furthermore|11||
hence|4||4
however|485||
if|1223||1
if and when|3||
if then|38||
in addition|165||
in contrast|12||
in fact|82||
in other words|17||
in particular|15||
in short|4||
in sum|2||
in the end|9||1
in turn|30||7
indeed|104||1
insofar as|1|1|
instead|112||
later|91||
lest|2||
likewise|8||
meantime|15||
meanwhile|193||
moreover|101||
much as|6||
neither nor|3||
nevertheless|44||
next|7||
nonetheless|27||
nor|31||
now that|22|18|2
on the contrary|4||
on the one hand on the other hand|1||
on the other hand|37||
once|84|7|1
or|98||
otherwise|24||
overall|12||
plus|1||
previously|49||
rather|17||
regardless|2||
separately|74||
similarly|18||
simultaneously|6||
since|184|104|
so|263|1|262
so that|31||31
specifically|10||
still|190||
then|340|2|12
thereafter|11||
thereby|12||12
therefore|26||26
though|320||
thus|112||112
till|3||
ultimately|18|1|2
unless|95||
until|162||
when|989|111|3
when and if|1||
whereas|5||
while|781||
yet|101||
'''

'''
as|743|334|2
because|858|854|
for|3|3|
insofar as|1|1|
now that|22|18|2
since|184|104|
'''

reason_markers = set(tuple(i.split()) for i in '''because'''.split('\n'))

'''
accordingly|5||5
as a result|78||78
consequently|10||10
hence|4||4
so|263|1|262
so that|31||31
thereby|12||12
therefore|26||26
thus|112||112
'''
#in response
#subsequently

result_markers = set(tuple(i.split()) for i in '''accordingly
as a consequence
as a result
consequently
hence
in response
so IN
so that
subsequently
thereby
therefore
thus'''.split('\n')) 

'''
after|577|50|
and|2999||177
as|743|334|2
finally|32||2
for|3|3|
in the end|9||1
in turn|30||7
insofar as|1|1|
now that|22|18|2
once|84|7|1
since|184|104|
so|263|1|262
so that|31||31
then|340|2|12
thereby|12||12
therefore|26||26
thus|112||112
ultimately|18|1|2
when|989|111|3
'''

possible_reason_markers = set(tuple(i.split()) for i in '''after
as
for
insofar as
now that
once
since
so
then
ultimately
when'''.split('\n'))

possible_result_markers = set(tuple(i.split()) for i in '''and
as
finally
in the end
in turn
now that
once
so
then
ultimately
when'''.split('\n'))


causal_markers = reason_markers | result_markers

'''
additionally|7||
afterward|11||
also|1746||
alternatively|6||
although|328||
as an alternative|2||
as if|16||
as long as|24||
as soon as|20||
as though|5||
as well|6||
before|326||
before and after|1||
besides|19||
but|3308|1|1
by comparison|11||
by contrast|27||
conversely|2||
earlier|15||
either or|4||
else|1||
except|10||
for example|196||
for instance|98||
further|9||
furthermore|11||
however|485||
if|1223||1
if and when|3||
if then|38||
in addition|165||
in contrast|12||
in fact|82||
in other words|17||
in particular|15||
in short|4||
in sum|2||
indeed|104||1
instead|112||
later|91||
lest|2||
likewise|8||
meantime|15||
meanwhile|193||
moreover|101||
much as|6||
neither nor|3||
nevertheless|44||
next|7||
nonetheless|27||
nor|31||
on the contrary|4||
on the one hand on the other hand|1||
on the other hand|37||
or|98||
otherwise|24||
overall|12||
plus|1||
previously|49||
rather|17||
regardless|2||
separately|74||
similarly|18||
simultaneously|6||
specifically|10||
still|190||
thereafter|11||
though|320||
till|3||
unless|95||
until|162||
when and if|1||
whereas|5||
while|781||
yet|101||
'''

noncausal_markers = set(tuple(i.split()) for i in '''additionally
afterward
also
alternatively
although
as an alternative
as if
as long as
as soon as
as though
as well
before
before and after
besides
but
by comparison
by contrast
conversely
earlier
either or
except
for example
for instance
further
furthermore
however
if
if and when
if then
in addition
in contrast
in fact
in other words
in particular
in short
in sum
in the end
indeed
instead
later
lest
likewise
meantime
meanwhile
moreover
much as
neither nor
nevertheless
next
nonetheless
nor
on the contrary
on the other hand
or
otherwise
overall
plus
previously
rather
regardless
separately
similarly
simultaneously
specifically
still
thereafter
though
till
unless
until
whereas
when and if
while
yet'''.split('\n'))

'''
after|577|50|
and|2999||177
as|743|334|2
finally|32||2
in turn|30||7
now that|22|18|2
once|84|7|1
since|184|104|
then|340|2|12
ultimately|18|1|2
when|989|111|3
'''

possible_noncausal_markers = set(tuple(i.split()) for i in '''after
and
as
finally
in turn
now that
once
since
then
ultimately
when'''.split('\n'))

all_markers = set(tuple(i.split()) for i in '''accordingly
additionally
after
afterwards
also
although
and
as
as a consequence
as a matter of fact
as a result
as it turns out
at that time
at the same time
at the time
because
before
besides
but
by comparison
by contrast
consequently
earlier
even though
eventually
ever since
finally
first
for
for example
for instance
for one
for one thing
further
furthermore
hence
however
in addition
in comparison
in contrast
in fact
in other words
in particular
in response
in return
in short
in sum
in summary
in the end
in the meantime
in turn
inasmuch as
incidentally
indeed
insofar as
instead
later
likewise
meanwhile
moreover
nevertheless
next
nonetheless
now
on the contrary
on the one hand
on the other hand
on the whole
or
overall
particularly
plus
previously
rather
regardless
second
separately
similarly
simultaneously
since
since then
so
so far
so that
soon
specifically
still
subsequently
that be
then
thereafter
therefore
third
though
thus
to this end
ultimately
what be more
when
whereas
while
yet'''.split('\n'))

modal_auxiliary = set('''be
can
could
dare
do
have
may
might
must
need
ought
shall
should
will
would'''.split('\n'))

'''
raw_pdtb_counts =
(('as', 'a', 'matter', 'of', 'fact'), 1)                                                             
(('insofar', 'as'), 1)
(('when', 'and', 'if'), 1)
(('that', 'be'), 1)
(('additionally',), 1)
(('as', 'it', 'turns', 'out'), 1)
(('if', 'then'), 1)
(('lest',), 2)
(('alternatively',), 2)
(('furthermore',), 2)
(('whereas',), 3)
(('in', 'other', 'words'), 3)
(('if', 'and', 'when'), 3)
(('before', 'and', 'after'), 4)
(('afterwards',), 4)
(('accordingly',), 4)
(('on', 'the', 'whole'), 4)
(('by', 'comparison'), 5)
(('likewise',), 5)
(('as', 'an', 'alternative'), 5)
(('consequently',), 5)
(('till',), 6)
(('at', 'that', 'time'), 7)
(('moreover',), 7)
(('nevertheless',), 7)
(('in', 'comparison'), 8)
(('afterward',), 9)
(('hence',), 9)
(('by', 'contrast'), 9)
(('in', 'the', 'meantime'), 9)
(('in', 'contrast'), 9)
(('ever', 'since'), 9)
(('in', 'short'), 10)
(('similarly',), 10)
(('besides',), 11)
(('subsequently',), 12)
(('as', 'though'), 13)
(('in', 'the', 'end'), 13)
(('since', 'then'), 16)
(('simultaneously',), 17)
(('nonetheless',), 17)
(('regardless',), 17)
(('thereby',), 17)
(('thereafter',), 18)
(('on', 'the', 'other', 'hand'), 19)
(('meantime',), 20)
(('in', 'return'), 22)
(('in', 'addition'), 23)
(('as', 'if'), 24)
(('in', 'particular'), 26)
(('specifically',), 29)
(('as', 'soon', 'as'), 30)
(('separately',), 30)
(('therefore',), 31)
(('at', 'the', 'same', 'time'), 31)
(('in', 'turn'), 33)
(('otherwise',), 33)
(('as', 'long', 'as'), 36)
(('so', 'that'), 37)
(('in', 'fact'), 38)
(('in', 'response'), 41)
(('at', 'the', 'time'), 43)
(('for', 'one'), 47)
(('ultimately',), 49)
(('indeed',), 56)
(('meanwhile',), 60)
(('except',), 64)
(('finally',), 67)
(('thus',), 67)
(('for', 'instance'), 73)
(('nor',), 74)
(('plus',), 74)
(('eventually',), 77)
(('unless',), 90)
(('as', 'a', 'result'), 92)
(('overall',), 93)
(('so', 'far'), 93)
(('even', 'though'), 94)
(('instead',), 140)
(('for', 'example'), 169)
(('although',), 170)
(('particularly',), 178)
(('previously',), 179)
(('rather',), 201)
(('much', 'as'), 201)
(('soon',), 203)
(('yet',), 244)
(('as', 'well'), 276)
(('later',), 277)
(('though',), 330)
(('further',), 340)
(('second',), 355)
(('however',), 361)
(('until',), 384)
(('then',), 466)
(('third',), 471)
(('since',), 639)
(('while',), 664)
(('still',), 669)
(('earlier',), 718)
(('before',), 744)
(('next',), 784)
(('so',), 855)
(('now',), 906)
(('if',), 1085)
(('first',), 1145)
(('after',), 1283)
(('when',), 1315)
(('because',), 1378)
(('also',), 1937)
(('but',), 2253)
(('or',), 3398)
(('as',), 5776)
(('for',), 10812)
(('and',), 21630)
'''

binaryCausalSettings = ({'causal': [set(i) for i in causal_markers],
                         'notcausal': [set(i) for i in noncausal_markers]},
                        {'notcausal' : 0,
                         'causal' : 1,
                         'other': 2})

trinaryCausalSettings = ({'reason': [set(i) for i in reason_markers],
                          'result': [set(i) for i in result_markers],
                          'notcausal': [set(i) for i in noncausal_markers]},
                         {'notcausal': 0,
                          'reason' : 1,
                          'result' : 2,
                          'other' : 3})

