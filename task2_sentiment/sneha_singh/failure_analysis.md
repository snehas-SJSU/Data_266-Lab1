# Part 2 — Error review (Sneha)

Model used: **baseline** (full)

| id | slice | gold | pred | confidence | error_type | snippet | testable_fix |
|---|---|---|---|---|---|---|---|
| 1 | confident_FP | 0 | 1 | 1.0 | false_positive | food always good | balance length / more epochs |
| 2 | confident_FP | 0 | 1 | 0.9998 | false_positive | good beer | balance length / more epochs |
| 3 | confident_FP | 0 | 1 | 0.9996 | false_positive | found hair sub professional refunded nice friendly staff | balance length / more epochs |
| 4 | confident_FP | 0 | 1 | 0.9992 | false_positive | house margs good cheap big like men | balance length / more epochs |
| 5 | confident_FP | 0 | 1 | 0.999 | false_positive | closed loved place sandwich shop north la vega offered freshly made sandwich people friendly sad | balance length / more epochs |
| 6 | confident_FN | 1 | 0 | 0.0 | false_negative | margarita wont ya dirty | balance length / more epochs |
| 7 | confident_FN | 1 | 0 | 0.0 | false_negative | not restaurant closed | add negation handling |
| 8 | confident_FN | 1 | 0 | 0.0 | false_negative | sago gulaman pretty good not sweet not bland right | add negation handling |
| 9 | confident_FN | 1 | 0 | 0.0001 | false_negative | decent | balance length / more epochs |
| 10 | confident_FN | 1 | 0 | 0.0002 | false_negative | closed due emergency sign said closed good nneed refund 20 voucher not want anyone else also inconvenienced | add negation handling |
| 11 | near_threshold | 0 | 1 | 0.5003 | false_positive | came week night place not packed like reviewer mentioned service slow one guy handling every table definitely waited long time even saw nthe side little quantit | add negation handling |
| 12 | near_threshold | 0 | 1 | 0.5005 | false_positive | disappointed went early morning wanting sit joy coffee coffee warm not good no comfortable relaxing place sit music way loud early day left without drinking cof | add negation handling |
| 13 | near_threshold | 0 | 1 | 0.5014 | false_positive | wow people really full nwe first learned gallery year ago another trip scottsdale boyfriend interested art selection remembered year located scottsdale road hea | add negation handling |
| 14 | near_threshold | 1 | 0 | 0.4971 | false_negative | place literally 50 foot work noodle ordered amazing quick prepare meal take earned customer back lunch next time service good offered water waited food even tak | balance length / more epochs |
| 15 | near_threshold | 0 | 1 | 0.5033 | false_positive | wife asking go kneaders since moved phoenix shocked small portion even shocked heard total cost meal place expensive ni guess pay grandma attic hoarder atmosphe | add negation handling |
| 16 | slice_specific_long | 0 | 1 | 0.5147 | false_positive | say found dress loved price range return customer service ni suppose could gotten get dress want negative experience taint wedding going well people helping u e | add negation handling |
| 17 | slice_specific_long | 1 | 0 | 0.1992 | false_negative | thoroughly impressed airport nnot great world traveler reason see booked virgin transatlantic flight yes virgin lower case popping passport cherry not capital g | add negation handling |
| 18 | slice_specific_long | 1 | 0 | 0.4549 | false_negative | summer time supposed healthy time trying watch eat always big guy not two fat twin matching motorcycle big know not little definitely eating appropriately howev | add negation handling |
| 19 | slice_specific_long | 0 | 1 | 0.9154 | false_positive | geee ross half got take place first little back story dad whole family worked chinese restaurant philly born late 1990 mother say craved chinese food whole time | add negation handling |
| 20 | slice_specific_long | 0 | 1 | 0.7196 | false_positive | opinion md wear three piece suit see doctor wearing three piece suit first thing come mind wearing suit wear suit class med school wear suit round residency kno | add negation handling |
