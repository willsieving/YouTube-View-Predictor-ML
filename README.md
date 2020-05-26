YouTube-View-Predictor-ML

This project was an attempt to create a Multiple Variable Linear Regression algorithm
that predicts the views on a YouTube video given the Likes, Dislikes, and Comments

Data was taken from over 600 YouTube channels to train the model.
This project was partially successful.

It can predict the views but not as accurately as I'd like.
With a range to cover 2σ (95% coverage) the actual views is always near the lower bound

Things to think on:
- Use more data, such as subscriber count and number of videos on channel.
- Generalizing all videos on YouTube is difficult, not all videos are equal.
- The model works better with more mainstream videos, such as the Trending section, and older videos.
- The model works best with mainstream older videos (a few years old)
- More testing needs to be done.

How to Use:
- Download 'Youtube Valuation Tool' and 'youtube.csv' into your directory
- Make sure the proper modules are installed
- import 'get_views_estimate' into your project
- Read docstring for arguments
