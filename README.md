# Price predictability at ultra high frequency

Python codes available to reproduce results from article "Price predictability at ultra-high frequency: Entropy-based randomness test"

Python files starting from "simulation" provide codes for predictability of models (_lambda, _OD, _TS) and QQ plots: 

_entropy for the test statistics based on Entropy, 

_KL_equal for testing NP-statistics when all symbols have equal probabilities, 

_KL_unequal when symbol 0 is more probable.

The rest scripts are to work with real data:

KL_blocklength to dispay test statistics together with a confident bound for AAPL stock on 02.08.22,

Entropy_and_KL_months to test the predicbility for different months,

Entropy_and_KL_nanoseconds to test the predictability for the limit order book aggreagated to nanoseconds,

statistics_predictable_days to test the difference in characterstics between predictable and not predictable days.
