# Correlation-between-global-events-and-films-production
A script containing a DNN showing a correlation between sociopolitical events around the world and predominant filmic genres in that year

Before I talk about the actual code or implementation, I think it is worthy to explain the motivation of this work.

It is well known that movies usually work as a representation of the current state of the society and even its deepest fears. For example, as William Tsutsui says in "Godzilla on My Mind: Fifty Years of the King of Monsters", Godzilla was a representation of the japanese people's fear towards the atomic bomb and the life conditions they had to endure during the occupation of their territory on the post-war period. Another instance is explained in "The Idle Proletariat:Dawn of the Dead, Consumer Ideology, and the Loss of Productive Labor" by Kyle William Bishop, where he propose that zombies movies are a representation of capitalism and consumerism driving humans to become entities to eat and destroy whatever and whoever they find.

The question here is: if some movie genres, for example horror, have the ability to represent the common fears on the society, why does not all type of topics are found on cinemas? For example, even if it is common to find all over the internet information and publications about racial inequality and discrimination towards the lgbt+ community or women, it is very rare to see a new horror film about this topics. I am not saying there are none, because one can find movies like Raw (2016) being a metaphore about the repression a girl suffers from her family or Black Christmas (2019) where a group of women takes down a cult of men which had kidnapped a friend of them, but on the other hand, there are 36 Godzilla movies...

Noticing this, I was interested in studying which kind of social, economic or political events actually have a repercution in the filmic industry. In order to do this, I downloaded the oen-access database from GDELT Project (https://www.gdeltproject.org/) which downloads daily online newspapers from all over the world and, using a neural network capable of processing natural language and emotions recognition, classifies the types of news into 20 categories using the CAMEO guidelines (available at http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf). This datasets contain news classification since 1979. The information I used from this database was the year of publication and the classification of the new. 
For further explanation of the meaning of the classification you can look the linked placed above, but the main possibilities for a new were: Make public statement, Appeal, Express intent to cooperate, Consult, Engage in diplomatic cooperation, Engage in material cooperation, Provide aid, Yield, Investigate, Demand, Disapprove, Reject, Threaten, Protest, Exhibit force posture, Reduce relations, Coerce, Assault, Fight, Use unconventional mass violence.

Once I had a good representation for the events across the world, I needed information about the produced movies across the time, so I took the IMDB movies database. What I used from this database was the releasing year of the movie and its genres.

The rest of the procedure will be described on the code. It's important to say this does not pretend to be a causal model, but just to be a correlation model in order to try to understand which types of events on the world tend to have more representation or impact into the cinematographic industry.
