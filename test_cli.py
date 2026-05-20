import sys, os; sys.path.insert(0, os.path.abspath('Responses/Src')); os.chdir('Responses/Src'); from query_engine import QueryEngine; print(QueryEngine().process_query('fire emergency'))
