# NASA Competition - Run-way Functions: Predict Reconfigurations at US Airports
*(https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/)*

Team Participants: Yue Li, Ellis Thompson, Shulu Chen

## Overview
The NAS is arguably the most complex transportation system in the world. Operational changes can save or cost airlines, taxpayers, consumers, and the economy at large thousands to millions of dollars on a regular basis. It is critical that decisions to change procedures are done with as much lead time and certainty as possible. The NAS is investing in new ways to bring vast amounts of data together with state-of-the-art machine learning to improve air travel for everyone.

An important part of this equation is airport configuration, the combination of runways used for arrivals and departures and the flow direction on those runways. For example, one configuration may use a set of runways in a north-to-south flow (or just "south flow") while another uses south-to-north flow ("north flow"). Air traffic officials may change an airport configuration depending on weather, traffic, or other inputs.

These changes can result in delays to flights, which may have to alter their flight paths well ahead of reaching the airport to get into the correct alignment or enter holding patterns in the air as the flows are altered. The decisions to change the airport configuration are driven by data and observations, meaning it is possible to predict these changes in advance and give flight operators time to adjust schedules in order to reduce delays and wasted fuel.

## Task
This task focuses on 10 major airports in the US. The goal of this challenge is to automatically predict airport configuration changes from real-time data sources including air traffic and weather. By using these data of different airports, models were built to predict future runway configruation from 30mins to 6 hours.

Better algorithms for predicting future airport configurations can support critical decisions, reduce costs, conserve energy, and mitigate delays across the national airspace network. 

![image](https://drivendata-public-assets.s3.amazonaws.com/airportconfig-airport-map.svg)

## Files
### 'katl exp' 
this folder contains initial experiments of katl. 
- data_process.py: select train dataset for 30 mins prediction. 
- katl-data process.py: select train dataset for 30 mins to 6 hours prediction.
- katl-models.py: train models for 30 mins to 6 hours prediction.
- 'xgboost test' folder: code for test the xgboost method on katl dataset.

### 'src' 
- data_processing_30mins.py: pepare 30 mins lookahead prediction training dataset of 10 airports 
- timebasedProcessing.py: pepare 30 mins to 6 hours lookahead prediction training dataset of 10 airports 
- xg_boost.py: xgboost method implement 

### 'Open Submission' 
'Code' folder: 
- generate test dataset based on required timestamp of this competition(generate_training_dataset.py)
- train models(train models.py) 
- make prediction(main.py)

## Rankings
The final mean aggregated log loss of our models was 0.0746. Our group ranked 9 in this competition. 
