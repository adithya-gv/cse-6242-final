CSE 6242 Final Project: A Video Game Recommender
By: Jackson Dike, Rishabh Jain, Ted Park, and Adithya Vasudev

Description: 
- This project is a video game recommendation system, that uses a proprietary metric called the Game Value Index, to determine 
  the best value games for a given user, based on their preferences. The system uses a KMeans clustering algorithm, combined with a 
  variant of the nearest neighbors algorithm, to determine the best games for a user.
- The Game Value Index is the unique marker of our project. It aims to quantify a wide variety of factors that go into a game's 
  quality, such as the duration, rating, and popularity, and combines them into a single metric, comparing it to other games  
  that released in the same year. This GVI score is used to feed our recommendation system to make its prediction.
- The package is well organized into three major sections:
    - The data folder contains a cache of a subset of the data used for the project. This is primarily for quickly populating the 
      visualization.
    - The dataGather folder contains the code used to gather data, both via webscraping and the API call. It also contains a local copy 
      of the nintendeals library, used to scrape the Nintendo eShop, as the library is no longer maintained and required a fix to get it to work.
    - The kmeans.py file contains the code to run live iterations of the kmeans clustering algorithm.
    - app.py contains the frontend logic for the application, and utils.py is the backend logic

Installation Intructions:
1. Clone the repository into a folder.
2. Run the command 'pip install -r requirements.txt' to install the relevant packages from the requirements file.
3. To spin up the application, run 'python app.py'

Demo Instructions:
1. To run a demo on our application, simply spin it up according to the installation instructions.
2. Open the app at the designated port.
3. The clustering will run automatically, populating the dashboard. Simply play with the fields as you like, and let the recommender suggest games for you!