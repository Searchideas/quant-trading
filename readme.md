### Explaination of factor based model buiolt

#### How to run 
1. python main.py --market us
2. python main.py --market sg --skip-download
3. python -m analytics.gridsearch --market us --step 0.1
4. pip install tigeropen
5. Copy broker/tiger_openapi_config.properties.example to tiger_openapi_config.properties and fill in the paper account
6. python -m broker --market us
7. python -m broker --market us --place

#### Pipeline
- Download via yahoofinance required data to compute factor
- Run factor value and retrieve sector mean and sd 
- Gridsearch can help to find the best sharpe ratio allocation
- Read the output file  
