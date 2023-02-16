cd "H:\OneDrive - Massachusetts Institute of Technology\Thesis\SimpleCode_to_LAMP_LSTM\SimpleCode_and_LAMP_generator"

:: set VAR_1=MED_expanded_training.txt
set VAR_2=MED_expanded_validation2.txt
set VAR_3=MED_expanded_test2.txt


:: echo %VAR_1% | python SimpleCodePy.py
echo %VAR_2% | python LAMPPy.py
echo %VAR_3% | python LAMPPy.py

@REM echo %VAR_1% | python SimpleCodePy.py
@REM echo %VAR_2% | python SimpleCodePy.py
@REM echo %VAR_3% | python SimpleCodePy.py

:: cd "H:\OneDrive - Massachusetts Institute of Technology\Thesis\SimpleCode_to_LAMP_LSTM"
:: python main.py