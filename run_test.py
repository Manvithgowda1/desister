import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'Responses', 'Src')
sys.path.insert(0, SRC)
os.chdir(SRC)

from query_engine import QueryEngine
qe = QueryEngine()
print(qe.process_query('fire emergency'))
