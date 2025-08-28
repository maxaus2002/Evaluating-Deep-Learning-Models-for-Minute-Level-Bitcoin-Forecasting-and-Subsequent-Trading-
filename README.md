# Evaluating-Deep-Learning-Models-for-Minute-Level-Bitcoin-Forecasting-and-Subsequent-Trading-
MAST7240 Final Project Max Austin

Report
	Contains the report explaining the project and all research

 Code
 	Contains the following


Bitcoin Forecasting & Algorithmic Trading

Two identically structured pipelines (LSTM.py and Transformer.py)
generate monthly forecasts and then run monthly or yearly-resetting
trading simulations on those forecasts.

Contents

-   Overview
-   Quick Start (easy)
- 	Quick Start (manual)
-   Configuration
-   Directory Structure
-   Forecasting Outputs
-   Trading Runs
-   Notes & Tips
-   Troubleshooting

------------------------------------------------------------------------

Overview

There are two main scripts and two follow-on scripts:

-   LSTM.py
-   Transformer.py

-   Algorithmic Based (Year-Year Portfolio)
-   Algorithmic Based (Month-Month Portfolio)

The two main scripts train their respective models and output forecasts
 The two follow-on scripts train their trading algorithms based on the respective forecasts 
(month-month is based on Monthly_forecasts, year-year is based on Monthly_forecasts_yearly_portfolio. shown below)

Forecasts are saved to two parallel output roots: -
monthly_forecasts/... (primary; also stores model + scalers) -
monthly_forecasts_Yearly_portfolio/... (mirror for “Yearly” portfolio
analyses)

------------------------------------------------------------------------
Quick Start - easy method

1) Ensure 7-Zip is installed on your system
	Can be downloaded from the following link: https://www.7-zip.org/download.html

2)  Choose a forecasting model 

	Transformer Model
    # or
	LSTM Model:		

3) Download the respective Zips from the following Github links 	(download .7z.001 - .7z.007)

	Transformer	https://github.com/maxaus2002/Evaluating-Deep-Learning-Models-for-Minute-Level-Bitcoin-Forecasting-and-Subsequent-Trading-/releases/tag/Forecasting_Transformer

	LSTM:		https://github.com/maxaus2002/Evaluating-Deep-Learning-Models-for-Minute-Level-Bitcoin-Forecasting-and-Subsequent-Trading-/releases/tag/Forecasting	


4) Move all downloaded zips to an empty folder

5) Highlight all -> right click -> 7zip -> Extract here

6) Download starting_year.txt from the same location as the model and move to the same folder as the resulting .exe from above

optional) The year in the .txt document Starting_year is the year that will be forecast from; by default, it is set to 2024. The earliest year possible is 2012. Set to None for it to do all years. 
	(Note - whatever year it is set to, the first 3 months will be used for training the lstm, and the 4th month for teaching the initial trading algorithm. And so all equity graphs will start from the 5th month.

7) Run the respective .exe

	Transformer.exe
    # or
	LSTM.exe

8) Repeat steps 3-5 for the desired trading algorithm (both can be ran, Algorithmic trading(month-month) will update the monthly_forecasts folder, Algorithmic trading(year-year) will update Monthly_forecasts_yearly_portfolio)

	Algorithmic trading(month-month)	https://github.com/maxaus2002/Evaluating-Deep-Learning-Models-for-Minute-Level-Bitcoin-Forecasting-and-Subsequent-Trading-/releases/tag/Algorithmic_Trading_Monthly_Portfolio

	
	Algorithmic trading(year-year)		https://github.com/maxaus2002/Evaluating-Deep-Learning-Models-for-Minute-Level-Bitcoin-Forecasting-and-Subsequent-Trading-/releases/tag/Trading

9)  Run the desired trading .exe
    -   Monthly-resetting:

	Algorithmic Based (Year-Year Portfolio).exe

    -   Yearly-resetting:

	Algorithmic Based (Month-Month Portfolio.exe

10)  Once completed, all equity graphs are viewable in the respective month folder
------------------------------------------------------------------------


Quick Start - Manual method

1) Ensure all relevant packages are installed as shown in the report, with all matching versions

2)  Choose a forecasting model 

	Transformer Model
    # or
	LSTM Model
3) Open the respective forecasting code
	Transformer.py
    # or
	LSTM.py

4)  Set your forecast span
    -   At the top of the configuration area, choose which year to start at

5) run forecasting code till all desired months have been forecasted or the code stops running

6)  Run trading 
    -   Monthly-resetting:

	Algorithmic Based (Year-Year Portfolio)

    -   Yearly-resetting:

	Algorithmic Based (Month-Month Portfolio

7)  Once completed, all equity graphs are viewable in their respective month folder
------------------------------------------------------------------------

Configuration

At the top of each script:

    # Forecast span control
    SKIP_BEFORE_YEAR = 2025   # set to an int year to start from that year
                              # set to None to include all years

-   Default: only forecast months in 2025+.
-   Change to another year (e.g., 2023) to start at that year.
-   Use None to process all available years in the dataset.

------------------------------------------------------------------------

Directory Structure

Both scripts write identical substructures under two roots:

    monthly_forecasts/
      └─ YEAR/
         └─ MONTH_NAME_YEAR/
            ├─ forecast_MONTH_YEAR.csv
            ├─ performance_MONTH_YEAR.png
            └─ (model.pt, scalers.pkl for the primary root only)

    monthly_forecasts_Yearly_portfolio/
      └─ YEAR/
         └─ MONTH_NAME_YEAR/
            ├─ forecast_MONTH_YEAR.csv
            └─ performance_MONTH_YEAR.png

  Metrics summaries (metrics_summary.csv) are generated in the root and
  appended per month.
  Only the primary root (monthly_forecasts) stores model.pt and
  scalers.pkl.

------------------------------------------------------------------------

Forecasting Outputs

For each forecasted month: - CSV: forecast_MONTH_YEAR.csv containing: -
datetime, close, fee fields - Multi-horizon predictions: forecast_t+Xm
for each horizon - Corresponding actuals: actual_t+Xm (where
available) - Plot: performance_MONTH_YEAR.png comparing average
predicted vs. average actual per horizon. - Metrics: Appended to
metrics_summary.csv at the root.

Outputs are written to: - monthly_forecasts/YEAR/MONTH_NAME_YEAR/ -
monthly_forecasts_Yearly_portfolio/YEAR/MONTH_NAME_YEAR/ (duplicate of
CSV + plot)

------------------------------------------------------------------------

Trading Runs

After forecasting the desired months:

-   Algorithmic_Trading (Monthly)
    Consumes monthly forecasts and writes trading artefacts back into:

        monthly_forecasts/YEAR/MONTH_NAME_YEAR/

    Produces:

    -   Best parameters, portfolio states, monthly equity graph, etc.

-   Algorithmic_Trading_Yearly_Portfolio (Yearly)
    Consumes the yearly portfolio forecasts and writes artefacts into:

        monthly_forecasts_Yearly_portfolio/YEAR/MONTH_NAME_YEAR/

    Produces:

    -   Monthly equity graphs (per month)

    -   Year-level equity graph for each year at:

            monthly_forecasts_Yearly_portfolio/YEAR/

------------------------------------------------------------------------

Notes & Tips

-   The two roots keep monthly and yearly portfolio analyses separate
    but aligned.
-   If you only care about forecasts and plots (and not copying
    models/scalers), duplication to monthly_forecasts_Yearly_portfolio
    already handles that.
-   If you later decide to store model artefacts in the yearly root,
    mirror the torch.save(...) and scaler pickle.

------------------------------------------------------------------------

Troubleshooting

-   No data written for earlier years
    	Check SKIP_BEFORE_YEAR. Set to a lower year or None.

- "RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread"
	Rerun Code, the issue happens very rarely during cache cleaning
