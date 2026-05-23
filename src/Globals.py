GENRE_SHELVES = {
    'fiction', 'non-fiction', 'nonfiction', 'fantasy', 'science-fiction', 'sci-fi',
    'mystery', 'thriller', 'romance', 'horror', 'biography', 'history',
    'self-help', 'classics', 'young-adult', 'ya', 'children', 'poetry',
    'comics', 'graphic-novels', 'crime', 'adventure', 'philosophy', 'science'
}
GENRE_MAPPING = {
    # fiction (only clear fiction labels)
    'fiction': 'fiction',
    'realistic-fiction' : 'fiction',
    'general-fiction' : 'fiction',
    'literary-fiction': 'fiction',
    'adult' : 'fiction',
    'adult-fiction' : 'fiction',
    'contemporary': 'fiction',
    'literature': 'fiction',
    'adventure': 'fiction',
    'historical-fiction': 'fiction',

    'poetry': 'poetry',



    # fantasy
    'fantasy': 'fantasy',
    'urban-fantasy': 'fantasy',
    'paranormal': 'fantasy',
    'sci-fi-fantasy': 'fantasy',
    'supernatural': 'fantasy',
    'vampires': 'fantasy',

    # science fiction
    'science-fiction': 'sci-fi',
    'sci-fi': 'sci-fi',
    'scifi': 'sci-fi',
    'dystopia' : 'sci-fi',
    'dystopian': 'sci-fi',


    # mystery/thriller
    'mystery': 'mystery',
    'crime': 'mystery',
    'thriller': 'mystery',

    # romance
    'romance': 'romance',
    'historical-romance': 'romance',
    'drama': 'fiction',

    # horror
    'horror': 'horror',

    # young adult / children
    'young-adult': 'young-adult',
    'ya': 'young-adult',
    'teen': 'young-adult',
    'childrens': 'children',
    'children': 'children',

    # non fiction
    'non-fiction': 'non-fiction',
    'nonfiction': 'non-fiction',
    'astrology': 'non-fiction',
    'biography': 'non-fiction',
    'science': 'non-fiction',
    'science': 'non-fiction',

    # separate non-fiction subgenres
    'history': 'history',
    'historical': 'history',  # move from historical-fiction
    'philosophy': 'philosophy',

    # classics
    'classics': 'classics',
    'classic': 'classics',
    'clàssics': 'classics',
    'classic-literature': 'classics',

    'manga': 'manga-comics',
    'comics': 'manga-comics',
    'graphic-novels': 'manga-comics'





}