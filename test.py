import sys
import os
sys.path.append(os.path.join(os.path.abspath('.'), 'Responses', 'Src'))
os.chdir(os.path.join(os.path.abspath('.'), 'Responses', 'Src'))
from query_engine import QueryEngine
qe = QueryEngine()
print('Response:', qe.process_query('fire emergency'))
