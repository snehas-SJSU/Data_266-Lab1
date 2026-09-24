# Part 2 — Error review (Sneha)

Model used: **baseline** (full)

| id | slice | gold | pred | confidence | error_type | snippet | testable_fix |
|---|---|---|---|---|---|---|---|
| 1 | confident_FP | 0 | 1 | 0.9984 | false_positive | line huge try pilot travel center | balance length / more epochs |
| 2 | confident_FP | 0 | 1 | 0.9971 | false_positive | buffet vega least favorite | balance length / more epochs |
| 3 | confident_FP | 0 | 1 | 0.9957 | false_positive | know good thai curry far service good | balance length / more epochs |
| 4 | confident_FP | 0 | 1 | 0.9956 | false_positive | buffet awesome think bellagio buffet better upside drink beer wine latte coffee pretty good dessert selection highly recommend pumpkin banana gelato come back d | add negation handling |
| 5 | confident_FP | 0 | 1 | 0.9948 | false_positive | parking lot dumb line rediculous best time go wednesday raining | balance length / more epochs |
| 6 | confident_FN | 1 | 0 | 0.0002 | false_negative | service average amount food give money hard beat | balance length / more epochs |
| 7 | confident_FN | 1 | 0 | 0.0037 | false_negative | bit expensive since hotel bar staff rock sushi not bad | add negation handling |
| 8 | confident_FN | 1 | 0 | 0.0054 | false_negative | not use service told correction needed clothes would not look good could done take money preferred advise not work definitely back need something done | add negation handling |
| 9 | confident_FN | 1 | 0 | 0.0158 | false_negative | come pho vega look no not find better pho even strip | add negation handling |
| 10 | confident_FN | 1 | 0 | 0.0329 | false_negative | month ago grandma u goto coco yuk came visit told wife anything coco dennys etc funny suggesting chili black angus would bring coco time told wife done type pla | balance length / more epochs |
| 11 | near_threshold | 1 | 0 | 0.4998 | false_negative | not often individual would drive four half hour bowl something fabby case folk pan roast redonkulous condemned death would last supper poor husband getting burn | add negation handling |
| 12 | near_threshold | 0 | 1 | 0.5004 | false_positive | theme review dumplinghaus overpriced idea smart ndumplinghaus name implies offer variety chinese dumpling baozi larger type steamed dumpling thicker skin chines | add negation handling |
| 13 | near_threshold | 1 | 0 | 0.4996 | false_negative | many pub feel like give full came last week based recommendation friend studied abroad year ago went order fajitas burger bartender asked student said asked see | add negation handling |
| 14 | near_threshold | 0 | 1 | 0.5005 | false_positive | show bizarre tried storyline make sense point completely laughable dancing like high school cheerleader routine dancer amazing pole dancer hand best part nthe p | add negation handling |
| 15 | near_threshold | 1 | 0 | 0.4988 | false_negative | bien que u00e7a soit un endroit extr u00eamement touristique en u00eame temp il tellement de recoins cach u00e9s et qui valent la peine u00eatre u00e9couvert da | add negation handling |
| 16 | slice_specific_long | 1 | 0 | 0.2181 | false_negative | contrary review zero complaint service price getting tire service past year compared experience place like pep boy guy experienced know nalso one place not feel | add negation handling |
| 17 | slice_specific_long | 0 | 1 | 0.5064 | false_positive | born raised italian family pretty darn picky pasta gravy tomato sauce unknowing great grandmother sauce first remember never opened make sauce always using toma | add negation handling |
| 18 | slice_specific_long | 1 | 0 | 0.4115 | false_negative | port authority formerly known patransit pat operates fairly extensive network bus south hill light rail instead running school bus school district gave high sch | add negation handling |
| 19 | slice_specific_long | 0 | 1 | 0.6558 | false_positive | walmart closest one shopped year recently remodeled carry grocery well normal retail section ni say store worst customer service ever seen employee not interest | add negation handling |
| 20 | slice_specific_long | 1 | 0 | 0.1254 | false_negative | one occassion park not bad price see coupon going start getting nwe not spending much lately nwell least week hee hee anyway cut time short last trip not think  | add negation handling |
