def fleiss_kappa(ratings, n, k):
        '''
        Computes the Fleiss kappa measure for assessing the reliability of
        agreement between a fixed number n of raters when assigning categorical
        ratings to a number of items.

        Args:
            ratings: a list of (item, category)-ratings
            n: number of raters
            k: number of categories
        Returns:
            the Fleiss kappa score
                
        See also:
        http://en.wikipedia.org/wiki/Fleiss_kappa
        '''
        items = set()
        categories = set()
        n_ij = {}

        for i, c in ratings:
            items.add(i)
            categories.add(c)
            n_ij[(i,c)] = n_ij.get((i,c), 0) + 1
            
        N = len(items)

        p_j = {}
        for c in categories:
            p_j[c] = sum(n_ij.get((i,c), 0) for i in items) / (1.0*n*N)

        P_i = {}
        for i in items:
            P_i[i] = (sum(n_ij.get((i,c), 0)**2 for c in categories)-n) / (n*(n-1.0))

        P_bar = sum(P_i.values()) / (1.0*N)
        P_e_bar = sum(p_j[c]**2 for c in categories)

        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

        return kappa
