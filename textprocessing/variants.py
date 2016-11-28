import difflib
import glob
import math
import os
from collections import defaultdict
from itertools import chain

from nltk.tokenize import regexp_tokenize

from textauger.utils import log


def get_unique_tokens(docs, pattern=r'\w+'):
    """
    Given a collection of strings (usually a list), tokenize it and return unique tokens.

    Args:
        docs (List[str])
        pattern (Optional[str]): Regular expression pattern used to split a string into substrings. Defaults to r'\w+'.

    Returns:
        Set[str]: Returns a set of unique tokens from docs.

    """
    log.debug("Getting unique tokens")
    if isinstance(docs, str):
        docs = [docs]
    tokenized_docs = [regexp_tokenize(doc.lower(), pattern) for doc in docs]
    unique_tokens = set(chain.from_iterable(tokenized_docs))

    return unique_tokens


def close_matches(word, possibilities, n=20, cutoff=0.3, output_dir=None):
    """
    Function to return a list of the best "good enough" matches.

    Utilizes the get_close_matches function in difflib. It has the option to write such
    matches to a .txt file which can be used to build a dictionary later on.

    Args:
        word (str)
        possibilities (List[str]): List of possible words to match word to.
        n (int): The maximum number of close matches to return. Defaults to 20.
        cutoff (float): Cutoff for considereing a word a match based on the similarity.
            In the range [0, 1]. Defaults to 0.3.
        output_dir (Optional[str]): Directory where to save the n best matches.
            The file name is of the form: word.txt. Spaces are replaced by "_".
            Defaults to None, which means it's not going to write a file.

    Returns:
        List[str]: The closest n matches among the possibilities, most similar first.

    """
    if output_dir:
        if not isinstance(output_dir, str):
            log.error("Parameter output_dir {} is not a string".format(output_dir))
            raise TypeError('Directory path must be string')
        if not os.path.isdir(output_dir):
            raise ValueError(
                "Directory {} does not exists or is not a directory".format(output_dir))

    close_matches = difflib.get_close_matches(word, possibilities, n, cutoff)

    if output_dir:
        file_name = word.strip().replace(' ', '_')
        output_locs = os.path.join(output_dir, file_name + ".txt" )
        with open(output_locs, 'ab') as fout:
            for match in close_matches:
                fout.write("%s\n" % match)

    return close_matches


def _variants_from_file(filepath, word, variants_dict):
    """
    Helper function for `make_variants_dict`, opens variants file and
    adds variants to existing `variants_dict`.

    Returns:
        Dict mapping variant to canonical form.

    """
    log.debug("Loading variants from file {}".format(filepath))
    with open(filepath) as infile:
        for line in infile:
            line = line.strip().lower()
            for variant in line.split():
                variants_dict[variant] = unicode(word, 'utf-8')
    return variants_dict


def _get_variants_files(directory, filepattern):
    """
    Helper for make_variants_dict by finding and returning files for variants.
    """
    glob_pattern = os.path.join(directory, filepattern)
    variant_files = glob.glob(glob_pattern)
    return variant_files


def make_variants_dict(directory, filepattern="*.txt"):
    """
    Make dictionary mapping variants to canonical form from files in given directory.

    The file name should be the canonical form, and the variants should be listed within the file,
    one per line. Files are found using the `glob` library.

    Args:
        directory (str): Directory where variant files are located.
        filepattern (str): Pattern for files within directory. Defaults to ``'*.txt'``.
            For acceptable patterns, see first paragraph of
            https://docs.python.org/2/library/glob.html

    Returns:
        Dictionary mapping variant to canonical form.

    """
    log.debug("Creating variants dictionary")
    if not isinstance(directory, str):
        log.error("directory parameter {} is not a string".format(directory))
        raise TypeError('directory must be given as string')
    if not os.path.isdir(directory):
        raise ValueError("Directory {} does not exist or is not a directory".format(directory))

    variants_files = _get_variants_files(directory, filepattern)

    word_variants = {}
    for filename in variants_files:
        __, tail = os.path.split(filename)
        # strip extension from filename and handle multi-words,
        # for example, "credit_card" -> "credit card"
        word = tail.split(".")[0].replace("_", " ")
        word_variants = _variants_from_file(filename, word, word_variants)
    return word_variants


def find_variants(tokens, n=40, cutoff=0.9, min_len=3, report_step=1000):
    """
    Groups `tokens` into sets of similar strings, as a *helper* to
    finding variant spellings, abbreviations, etc.

    Iterates through `tokens`. At each step, selects a token, finds
    similar strings in remaining `tokens`, removes that group, and
    stores them. Returns a list of those sets of tokens, sorted from
    largest set to smallest.

    This function is meant to be possible first step in finding
    variants. It not meant to be an automatic full solution to the
    problem. Also, this function can take a long time to run if you
    have more than a couple thousand tokens.

    Args:
        tokens (Set[str]): the tokens (words) that you want to group into variants.
        n (int): max number of close matches to return in each search. Defaults to 40.
        cutoff (float): Cutoff for considering a word a match based on the similarity.
            In the range [0, 1]. Higher values are more restrictive. Defaults to 0.9.
        min_len (int): minimum length of token to find similar tokens for.
            Any smaller token will be ignored. Defaults to 3.
        report_step(Optional[int]): print update every `report_step` tokens removed.
            If None, no update is printed. Defaults to 1000.

    Returns:
        List[Set[str]]: Each set contains similar strings.

    """
    log.debug("Finding variants")
    if not isinstance(tokens, set):
        raise TypeError('tokens must be in a set')

    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    if not n > 0:
        raise ValueError('n must be positive')

    if not isinstance(cutoff, float):
        raise TypeError('cutoff must be a float')
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError('cutoff must be >= 0.0 and <= 1.0')

    if not isinstance(min_len, int):
        raise TypeError('min_len must be an integer')

    if report_step is not None:
        if not isinstance(report_step, int):
            raise TypeError('report_step must be an integer or None')
        if not report_step > 0:
            report_step = None


    if report_step is not None:
        report_val = report_step
        report_ndigits = int(math.ceil(math.log(report_step, 10))) + 2
        report_tmplt = "{{:>{n}}} seen, {{:>{n}}} remaining".format(n=report_ndigits)
        seen, n_seen = set(), 0

    tokens_copy = tokens.copy()
    variants = []

    while tokens_copy:
        token = tokens_copy.pop()
        if len(token) < min_len:
            continue
        matches = set(
            token
            for token in close_matches(token, tokens_copy, n=n, cutoff=cutoff)
            if len(token) >= min_len
        )
        if matches:
            tokens_copy.difference_update(matches)

            matches.add(token)
            variants.append(matches)

            if report_step is not None:
                seen.update(matches)
                n_seen += len(matches)
                if n_seen > report_val:
                    # print report_tmplt.format(n_seen, len(tokens_copy))
                    report_val += report_step

    variants.sort(key=lambda matches: len(matches), reverse=True)

    return variants


def find_duplicates(directory, filepattern="*.txt"):
    """
    Find duplicate tokens in variants files.

    Use this, for example, when creating your own variants files, and
    wanting to check that you haven't put tokens in multiple files.

    Args:
        directory(str): Directory where variant files are located.
        filepattern (str): Pattern for files within directory. Defaults to ``'*.txt'``.
            For acceptable patterns, see first paragraph of
            https://docs.python.org/2/library/glob.html

    Returns:
        Dictionary mapping duplicated variants to a list of filenames in
        which they appear. Non-duplicates will not appear in the dictionary.
        Thus an empty dictionary indicates no duplicates.

    """
    log.debug("Finding duplicate tokens")
    if not isinstance(directory, str):
        raise TypeError('directory must be given as string')
    if not os.path.isdir(directory):
        raise ValueError('given string is not a directory path')

    variants_files = _get_variants_files(directory, filepattern)

    all_variants = defaultdict(list)
    duplicates = {}
    for filepath in variants_files:
        new_variants = {}
        _, filename = os.path.split(filepath)
        _variants_from_file(filepath, filename, new_variants)
        for variant in new_variants:
            if variant in all_variants:
                duplicates[variant] = all_variants[variant]
            all_variants[variant].append(filename)
    return duplicates
