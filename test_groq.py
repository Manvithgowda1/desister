import sys
import os

sys.path.append(os.path.abspath('Responses/Src'))
from query_engine import QueryEngine
qe = QueryEngine()
print(qe.process_query('fire emergency'))
