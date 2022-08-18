cd "H:\OneDrive - Massachusetts Institute of Technology\Thesis\SimpleCode_to_LAMP_LSTM\SimpleCode_and_LAMP_generator"

::set VAR_1=wide_training.txt
::set VAR_2=wide_validation.txt
set VAR_3=par_test.txt

::echo %VAR_1% | python SimpleCodePy.py
::echo %VAR_2% | python SimpleCodePy.py
::echo %VAR_3% | python SimpleCodePy.py

::echo %VAR_1% | python LAMPPy.py
::echo %VAR_2% | python LAMPPy.py
echo %VAR_3% | python LAMPPy.py

:: cd "H:\OneDrive - Massachusetts Institute of Technology\Thesis\SimpleCode_to_LAMP_LSTM"
:: python main.py