# Part 2 — Error review (Sneha)

Model used: **experimental_a** (full)

| id | slice | gold | pred | confidence | error_type | snippet | testable_fix |
|---|---|---|---|---|---|---|---|
| 1 | confident_FP | 0 | 1 | 0.9989 | false_positive | wow love place everything clean new ngreat place come relax worth try ncheers neric van nguyen nvisited april 2012 | balance length / more epochs |
| 2 | confident_FP | 0 | 1 | 0.9975 | false_positive | love paris love vega version hotel sure eiffel tower montgolfier balloon arc de triomphe sure one favorite restaurant mon ami gabi terrific buffet sure room fac | add negation handling |
| 3 | confident_FP | 0 | 1 | 0.9967 | false_positive | today first visit chicken pad thai lunch special soup great pad thai delivered record time asked spicy like archi thai kitchen however note heat bordered sweet  | add negation handling |
| 4 | confident_FP | 0 | 1 | 0.9964 | false_positive | favorite band come town end playing clubhouse miike snow last night great show ni used configuration split 21 21 easy get front see everything close personal so | add negation handling |
| 5 | confident_FP | 0 | 1 | 0.9961 | false_positive | like clay posted love pancake though love chocolate chip pancake like clay not love one got original pancake house typically restaurant way like come expect gra | add negation handling |
| 6 | confident_FN | 1 | 0 | 0.0008 | false_negative | edit really change service since last posted nhorrible service nused favorite pizza city reasonable price rethinking altercation server refused split check payi | add negation handling |
| 7 | confident_FN | 1 | 0 | 0.001 | false_negative | look know cox suck fact terrible business practice laughable however necessary evil want internet garbage handle streaming gaming way handle issue phone worse c | add negation handling |
| 8 | confident_FN | 1 | 0 | 0.0011 | false_negative | sum nwe going legacy package 86 per person cashier told u want food package basic skywalk 76 people thats saving 60 already plus planned ahead brought sandwich  | add negation handling |
| 9 | confident_FN | 1 | 0 | 0.0011 | false_negative | place much better since changed owner nmy wife went old owner terrible waited forever food never came walked people served u walked wife actually got soup sat w | add negation handling |
| 10 | confident_FN | 1 | 0 | 0.0013 | false_negative | two time first time really good second boring flavored lemon chicken mediocre nfor price last time turned probably go back | balance length / more epochs |
| 11 | near_threshold | 0 | 1 | 0.5002 | false_positive | wow seems like taco bell arizona full beer wine liquor license | balance length / more epochs |
| 12 | near_threshold | 0 | 1 | 0.5004 | false_positive | yo came u00ectch whole squad aint serviced like couple hour homies set dam camp lobby cuz afraid find shelter sundown | add negation handling |
| 13 | near_threshold | 1 | 0 | 0.4996 | false_negative | wow stupid good amazing burger slider buffalo sammy truffle fry ask garlic aioli dont miss garlic chili wing one big arm come negative hmmm no real negative imp | add negation handling |
| 14 | near_threshold | 0 | 1 | 0.5005 | false_positive | spent midweek safety conference last day large amount attendee contracted norovirus became violently ill say however incident room service actually pretty good  | add negation handling |
| 15 | near_threshold | 1 | 0 | 0.4992 | false_negative | half order mashed potato omelet ice tea everyone start day | balance length / more epochs |
| 16 | slice_specific_long | 1 | 0 | 0.0121 | false_negative | given mixed review quite sure expect came recently fittingly say experience mixed although overall positive nwe went saturday evening thought place would hoppin | add negation handling |
| 17 | slice_specific_long | 1 | 0 | 0.0566 | false_negative | basic class speeding ticket removed hour went ssssssllllowwwlllllllllyyyyy cyclops read slow nanyhow no video animation lame nall end chapter question made fina | add negation handling |
| 18 | slice_specific_long | 0 | 1 | 0.5202 | false_positive | another big time fail yelp community ncraving serious breakfast long night partying already bookmarked restaurant prior la vega trip based stellar review ventur | add negation handling |
| 19 | slice_specific_long | 1 | 0 | 0.1037 | false_negative | recently bought mid century house no furniture couple room went rubin clearance center happened catch moving sale meant everything marked normal clearance price | add negation handling |
| 20 | slice_specific_long | 0 | 1 | 0.8812 | false_positive | much criticism compliment already detailed nicely others summarize bullet point n1 great location walker inconvenient driving neighborhood given geared toward n | add negation handling |
