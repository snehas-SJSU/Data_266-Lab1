# Task 2.2(4) — Manual Error Review (textcnn)

Worst-performing length slice on this model: **long**

## Case 1 — confident_false_positive
- true=0, pred=1, P(positive)=1.000, length=23 tokens
- text: "By far, the least impressive Cirque du Soleil show I have seen.  The performers are all amazing, but for some reason, it was not showcased or directed well.\n\nWe spent over $130 on each ticket, and we had excellent seats, which were very comfortable, But the show seem to drag on.  The sand artist was really cool and different."
- likely cause: sarcasm / negation not captured; strong positive surface words despite a negative review
- **proposed testable fix:** mean-pooling destroys word order, so 'was not showcased or directed well' contributes the same bag-of-words signal as 'showcased and directed well'; test: add bigram features spanning negation words (not/n't + next word) as extra vocab entries, or swap the baseline's mean-pool for the TextCNN's local n-gram filters, which should recover most of this class of error.

## Case 2 — confident_false_positive
- true=0, pred=1, P(positive)=1.000, length=36 tokens
- text: "Leonard. Leonard. Leonard.  Is there REALLY even a Leonard that works there?  Leonard goes to work and then takes an 8 hour break.  I called Staples to follow up on a print order and he told me he was going to have to call me back because it was time for his break.  I then went to the store and ole Leonard was too busy on his cell phone to help me.  The only upside to this place is the nice female that works in the copy/print center."
- likely cause: sarcasm / negation not captured; strong positive surface words despite a negative review
- **proposed testable fix:** the review is a string of sarcastic rhetorical questions ('Is there REALLY even a Leonard...') with no explicit negative-sentiment words at all; test: this needs discourse-level features (rhetorical-question detection, repeated-word emphasis) that none of these 3 from-scratch models have -- likely out of reach without a pretrained LM, so document as a known ceiling rather than a fixable preprocessing bug.

## Case 3 — confident_false_positive
- true=0, pred=1, P(positive)=1.000, length=200 tokens
- text: "Je suis une grande gourmande et amante de plusieurs restaurants Italiens. De BONS restaurants Italiens. En vantant \u00e0 un coll\u00e8gue de travail 2 de mes adresses ador\u00e9es de la P'tite Italie, ce dernier m'a vivement recommend\u00e9 d'essayer le Caf\u00e9 Epoca que je n'avais jamais os\u00e9 tester. \nChaque fois que je suis pass\u00e9e devant cet \u00e9tablissement, il \u00e9tait vide ou compl\u00e8tement vide. Je me suis oblig\u00e9e \u00e0 visiter donc Epoca par respect pour mon coll"
- likely cause: sarcasm / negation not captured; strong positive surface words despite a negative review
- **proposed testable fix:** this review is in French, not English -- the preprocessing pipeline's stopword list and Porter stemmer are English-only, so the cleaned tokens are mostly untouched non-English words the English-trained vocab never learned useful embeddings for; test: add a language-detection filter in preprocess.py (e.g. langdetect) and exclude/flag non-English reviews at data-loading time, since this is a data-quality issue, not a model-architecture one.

## Case 4 — confident_false_positive
- true=0, pred=1, P(positive)=1.000, length=33 tokens
- text: "Good prices, good-quality produce and a wide variety of Middle-Eastern and Latin American groceries, if you are looking for meats you'll have to go somewhere else. The Middle- Eastern guys who run the business could learn a bit of Western politeness (they are in the West, after all) and quit that aggressive attitude, more than once I have felt as I am picking groceries from a food bank rather than a place where I am paying for them."
- likely cause: sarcasm / negation not captured; strong positive surface words despite a negative review
- **proposed testable fix:** the review lists positive adjectives ('good prices, good-quality produce, wide variety') before a late-review pivot to negative complaints about staff attitude; test: check whether weighting later tokens more heavily (or truncating from the end instead of the start at max_len=200) changes the prediction, since the negative payload is second-half content.

## Case 5 — confident_false_positive
- true=0, pred=1, P(positive)=1.000, length=54 tokens
- text: "Attended the @SpaFitFinder launch party, with @spacephx.\n\nI can't say much about the place, other than, even taking into consideration the number of people present, it felt cramped.\n\nThe TrimTini (?) I had was really delicious; vodkatini with lemon zest.\n\nA very nice gentleman, who works at Urban 7, was very gracious in making sure to stop by every time he brought out a new platter of hors d'oeuvres. Everything he brought was delicious, including the ceviche, which I'm not particularly fon"
- likely cause: sarcasm / negation not captured; strong positive surface words despite a negative review
- **proposed testable fix:** heavy sarcasm/backhanded-compliment structure ('really delicious... which I'm not particularly fon[d of]') cut off mid-word by max_len truncation, which may itself be dropping the actual negation; test: rerun this exact case with max_len raised past 200 tokens to check whether truncation (not the model) is the proximate cause.

## Case 6 — confident_false_negative
- true=1, pred=0, P(positive)=0.000, length=177 tokens
- text: "Look, we all know Cox sucks. In fact they are a terrible business and their practices are laughable. However they are a necessary evil if you want Internet that isn't garbage that can handle streaming/gaming. The way they handle issues over the phone is worse than Comcast, and for all my east coast brothers/sisters you know what a serious accusation that is. It all started with shitty TV Service. My household went through 4 dvr boxes. 4. That all broke. All of them. 100's of hours of TV. Lost. E"
- likely cause: negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)
- **proposed testable fix:** opens with strong negative words ('Cox sucks', 'terrible', 'laughable') that get the highest surface weight even though the review pivots to a qualified positive verdict; test: same fix direction as case 1 -- contrastive-conjunction-aware features (weight tokens after 'however'/'but' more heavily) should directly target this pattern.

## Case 7 — confident_false_negative
- true=1, pred=0, P(positive)=0.000, length=44 tokens
- text: "No problems at check-in.  The suite was nice on the 31st floor.  The bedroom  was well equipped with 3 sinks in the bathroom  hot tub, a shower for 2 or 4. The bathroom was sealed off wirh a glass door. It had a living room.with a bathroom by the entrance of the door. \n\n The only complaint was the sinks drained to slow which turned out to overflow.  We had to call housekeeping to straighten out.  The plummet stated they made these sinks more for beauty than use. \n\nThey transferred us to anot"
- likely cause: negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)
- **proposed testable fix:** mostly neutral/positive descriptive content with one short 'only complaint' clause; test: check whether this is actually a labeling/ambiguity issue in the source data (P(positive)=0.000 is a very confident wrong call for a review this mild) -- worth spot-checking a sample of the Yelp Polarity label distribution for similarly borderline reviews before assuming it's purely a model failure.

## Case 8 — confident_false_negative
- true=1, pred=0, P(positive)=0.000, length=38 tokens
- text: "Not a club. A swanky bar straight out of Austin Powers.  Free entrance and drinks for ladies when we came on Saturday night. Can't hate on that. Music was good.  People from all walks of life started dancing.  And dancing with all of their might.\n\nI think staffers caught sight of  the hideous dancing, so they cut the music to almost inaudible. Booo.  \n\nNo dancing, just \""lounging' allowed."
- likely cause: negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)
- **proposed testable fix:** double negatives and colloquial approval phrasing ('Can't hate on that', 'Booo' as mock-disapproval of a positive thing) -- classic negation-of-negation the mean/local-window models can't resolve; test: same contrastive/negation-scope feature fix as case 1/6.

## Case 9 — confident_false_negative
- true=1, pred=0, P(positive)=0.000, length=12 tokens
- text: "They are CLOSED due to A/C emergency.  The sign said CLOSED FOR GOOD!  \nNeed a refund on our $20 voucher and do not want anyone else to be also inconvenienced."
- likely cause: negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)
- **proposed testable fix:** short review, factually negative-sounding statement ('CLOSED FOR GOOD') but labeled positive in the source data -- this reads as a genuinely ambiguous or possibly mislabeled example rather than a clean model error; test: manually verify the ground-truth label for this specific review before spending model-side effort on it.

## Case 10 — confident_false_negative
- true=1, pred=0, P(positive)=0.000, length=71 tokens
- text: "Last night several parents came in with over 15 children to celebrate their 9 year olds 4th grade graduation at 9:15 pm. The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue if anything were to happen to them on their premise. The children were running in and out of the bar while the parents continued to drink upstairs claiming to watch them in the open park way. After their third round of drinks the bartender told them th"
- likely cause: negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)
- **proposed testable fix:** long review builds context (family event, then a specific policy complaint) before a resolution; test: check whether the model's error rate on this error_type correlates with reviews where the sentiment-bearing clause is buried past the first ~100 tokens (truncation-position sensitivity), same class of fix as case 4.

## Case 11 — near_threshold
- true=0, pred=1, P(positive)=0.500, length=124 tokens
- text: "OK, I am not a shopper...by no means! But, as part of the activities for a small family get together in Vegas, I agreed to go along. I had no idea what to expect but had heard that there was this great new mall in vegas.\n\nLet me preface this slam of a review by saying it was the end of June in Vegas which means it was hot, but not yet Vegas (read: Africa) Hot yet. I think it was about 100-102.\n\nSo, the backers of this project must have picked their architect from a pool of guys passing aroun"
- likely cause: genuinely mixed / ambivalent review, or short review with too little signal
- **proposed testable fix:** prediction sits at P=0.500 -- essentially a coin flip -- on a review that is itself genuinely mixed (dislikes shopping, but went along with family; complains about heat); test: this is a calibration/abstention case, not a clear error -- worth reporting a 'defer below confidence margin e.g. 0.45-0.55' policy in the write-up rather than treating every near-threshold case as fixable.

## Case 12 — near_threshold
- true=1, pred=0, P(positive)=0.500, length=62 tokens
- text: "The employees at this Target seemed unusually friendly during my last visit. Two of them asked if they could help me find anything, the cashier was perky and personable, and the security guard offered to put my shopping cart back for me as I was walking out with my bags. Maybe I've gotten too used to poor customer service, no smiles, no eye contact, and mumbling from employees when shopping. But, yesterday's visit was refreshing. I wasn't the only one who seemed to enjoy this Target. There was a"
- likely cause: genuinely mixed / ambivalent review, or short review with too little signal
- **proposed testable fix:** positive review content (friendly staff) but P=0.499 -- essentially correct despite crossing the decision boundary the wrong way; test: same as case 11 -- report this as a calibration-margin case; also worth checking Brier score/ECE per model (already computed) to see if TextCNN specifically is under-confident near the boundary.

## Case 13 — near_threshold
- true=1, pred=0, P(positive)=0.500, length=33 tokens
- text: "We met some long lost friends here last night.  Fast and efficient service.  The only issue was that music was to start at 9:00pm...we left at 10:30 and the music still had not begun.  At least we were able to chat with our friends but we were a bit disappointed as the sound check sounded great.  The fish sliders were super!  Potatoes skins needed salt.  I'd certainly go back."
- likely cause: genuinely mixed / ambivalent review, or short review with too little signal
- **proposed testable fix:** mixed review (long wait for music, but positive overall) -- true ambiguity in the source text itself; no model-architecture fix expected to move this case specifically.

## Case 14 — near_threshold
- true=0, pred=1, P(positive)=0.500, length=200 tokens
- text: "I've been here twice. The first time, my husband and I were using a restaurant.com gift so we splurged. Their spring rolls are awesome and fresh. The dishes were full of flavor and you could really taste the herbs which were great. We enjoyed the thai tea and desserts. It was a first for both of us to try the bean desserts, super yummy.\n\nThe second time I went with my friends. We had an order of spring rolls and soup. Two of us got pho and the other friend had the spicy noodle soup. When my fr"
- likely cause: genuinely mixed / ambivalent review, or short review with too little signal
- **proposed testable fix:** positive food review but P=0.500 lands on the wrong side by the smallest possible margin; same abstention-policy note as case 11.

## Case 15 — near_threshold
- true=1, pred=0, P(positive)=0.500, length=48 tokens
- text: "DELISH!  \nIt's hard to find 'good Italian' west of NY.....\nThis place meets the craving for sure!\nThe 'complimentary cheesy bread' offers a warm welcome upon sitting down.\nThe food is reasonably priced and the serving sizes are great - not too big / not too small.\nThey always offer a little post-meal Sambuca.\nThey make you feel at home.\nThe place is small - but cozy and warm.  You truly feel like you are at an italian restuarant.\nWouldnt call it kid friendly due to the size.\nLove the me"
- likely cause: genuinely mixed / ambivalent review, or short review with too little signal
- **proposed testable fix:** strongly positive review text ('DELISH!', multiple explicit compliments) landing at P=0.500 is the most surprising near-threshold case here -- test: rerun this single example with the other two models (baseline, bilstm) to check whether it's a TextCNN-specific quirk (e.g. its 3/4/5-gram kernels missing 'DELISH' as an all-caps, non-lowercased-post-stem token) rather than a genuinely hard example.

## Case 16 — slice_specific
- true=0, pred=1, P(positive)=0.549, length=64 tokens
- text: "I used to love D&B when it first opened in the Waterfront, but it has gone down hill over the years. The games are not as fun and do not give you as many tickets and the prizes have gotten cheaper in quality. It takes a whole heck of a lot of tickets for you to even get a pencil! The atmosphere is okay but it used to be so much better with the funnest games and diverse groups of people! Now, it is run down and many of the games are app related games (Fruit Ninja) and 3D Experience rides. With su"
- likely cause: systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate
- **proposed testable fix:** 200-token review (hit the max_len truncation cap) with a late sentiment reversal ('gone down hill... funnest games... now it is run down'); test: since 'long' is the worst-performing slice, specifically test raising max_len (e.g. 300-400) for this slice or truncating from the tail instead of the head, then re-score only the 'long' bucket to isolate whether truncation position is the systematic cause.

## Case 17 — slice_specific
- true=1, pred=0, P(positive)=0.259, length=157 tokens
- text: "As a chinese american college student from NYC who got stuck in Pittsburgh with its useless public transportation (circa 1998), I desperately missed home-style Chinese food for the better part of my first two years here. Things turned around for me once I got a car (or made friends with cars), but I was seriously hurting for a while.\n\nTasty was one of the places we found. Back in the day, it was owned and ran by a typical restaurant-owning Chinese family. They had the works - pushy lao ban nia"
- likely cause: systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate
- **proposed testable fix:** another 200-token-adjacent review where the actual verdict phrase ('Tasty was one of the places we found') is well past the point a length-200 truncation may cut off; same truncation-position test as case 16.

## Case 18 — slice_specific
- true=0, pred=1, P(positive)=0.810, length=154 tokens
- text: "Being born and raised into an all Italian family, I'm pretty darn picky about my pastas and gravy (tomato sauce for the unknowing). My great grandmother's sauce was the first I remember, she never opened a can to make her sauces, always using tomatoes she canned herself in mason jars. She simmered her sauces until every seed melted away. My mother's sauce was a bit more thicker, using puree and paste as a base, later adding whole peeled tomatoes, garlic and fresh basil. Her sauce cooks down to a"
- likely cause: systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate
- **proposed testable fix:** long, meandering family-history narrative before any restaurant opinion appears; test: same truncation fix direction -- also worth testing whether a hierarchical/chunked encoding (average sentence-level pooled vectors instead of one flat 200-token sequence) helps specifically on this slice.

## Case 19 — slice_specific
- true=0, pred=1, P(positive)=0.518, length=200 tokens
- text: "Hotel restaurant/bars are usually not known for being the best in town.  So when I had the opportunity to eat and drink at SoHo recently, I can not say I was surprised that I found the cuisine to be on the uninspiring side.  Is it terrible?  No,  but you know that better is out there.  \n\nThe blackened chicken wrap was boring.  The flavor was missing and the entire meal was just bland.  Also, white tortillas just do not do it for me.  Why does it seem that Pittsburgh wants to thrive on fatty, f"
- likely cause: systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate
- **proposed testable fix:** explicit hedge-then-negative structure ('Is it terrible? No, but you know that better is out there... boring... bland') that a length-200 window may or may not fully capture depending on where the review's real content starts; same truncation-position test as case 16.

## Case 20 — slice_specific
- true=0, pred=1, P(positive)=0.628, length=99 tokens
- text: "Last weekend my mother and I headed down to the strip to get some weekly goods - lunch meat and cheese from Penn Mac - veggies at Stans - needed some new spices from Penzeys for a chili I was going to make that day....we were both a little hungry and knew we wanted to have a snack before shopping.\n\nWe parked in front of Roland's and see the sign: Famous for Lobster Rolls and Bloody Mary's!\n\nPerfect. \n\nMy mother got the Lobster Roll, which she said was 'good'. The bread, from Mancini's, was"
- likely cause: systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate
- **proposed testable fix:** long narrative review (shopping trip anecdote) with the actual product opinion buried in the final third; test: this is the fourth of five 'long' slice cases with the same failure shape (opinion arrives late, truncation likely cuts or under-weights it) -- strong enough pattern across cases 16/17/18/20 to prioritize the tail-truncation experiment over any other single fix for this slice.
