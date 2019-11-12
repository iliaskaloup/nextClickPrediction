# nextClickPrediction
Products' cluster recommendation using deep learning techniques

Τα αρχεία του repository ανεφέρονται στα αρχεία κώδικα σε γλώσσα Python που αναπτύχθηκαν κατά τη διάρκεια εκπόνησης της διπλωματικής μου εργασίας.
Υπάρχουν τα αρχεία για το pre-processing (importData.py, importRecSys.py, pharmPreprocessing.py) του καθενός dataset 
Υπάρχουν τα αρχεία για την εκπαίδευση των μοντέλων με βάση 4 datasets:
α) RecSys:
b) UCI:
γ) pharm24
δ) pharm24 by clusters.
Επίσης υπάρχει το αρχείο clusteringOnPharm που υλοποιεί το clustering των προϊόντων,
καθώς και τα αρχεία που περιέχουν τα data της εργασίας.
Το clickPrediction.py υλοποιεί την εκπαίδευση και την αξιολόγηση για το μικρό e-shop και 
ακολούθως το randomRecommendation την αξιολόγηση για random recommendation του μικρού e-shop.
To myRnn_Items.py υλοποιεί την εκπαίδευση και την αξιολόγηση για το μεγάλο e-shop.
Το pharmItemPredictionNadam.py υλοποιεί την εκπαίδευση και την αξιολόγηση για το μεσαίο e-shop-ηλεκτρονικό φαρμακείο.
Το fit_generator.py υλοποιεί την εκπαίδευση και την αξιολόγηση για το μεσαίο e-shop με βάση τις ομάδες.
Επίσης υπάρχουν τα αρχεία που προβλέπουν μόνο το last click του κάθε session, καθώς και οι δοκιμές αυτού του task με MLP.
Το ClustersPredictionFromPharm.py υπολογίζει την ακρίβεια να προβλεφθεί η σωστή ομάδα από την πρόβλεψη του επομενου προϊόντος, 
ενώ το pharmPredictionFromClusters.py υπολογίζει την ακρίβεια να προβλεφθεί το σωστό προϊόν από την πρόβλεψη της επόμενης ομάδας.
