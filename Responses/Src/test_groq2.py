import sys
import os
sys.path.append(os.path.abspath('.'))
from query_engine import QueryEngine
qe = QueryEngine()
print('Response:', qe.call_groq('fire emergency'))
