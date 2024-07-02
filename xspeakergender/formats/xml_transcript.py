# coding: utf-8

import argparse
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET

# We include the path of the toplevel package in the system path,
# so we can always use absolute imports within the package.
toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

import utils.constants as cst
import utils.util as utl


# READER  ##############################################################################################################

class TranscriptReader:
    def __init__(self, file_path):
        self.speech_segment_index = -1
        self.word_index = -1
        self.speech_segment_next_index = 0
        self.word_next_index = 0

        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.speakers = {speaker.attrib["spkid"]: speaker for speaker in self.root[2]}

        # There are empty speech segments sometimes
        while (self.speech_segment_next_index < len(self.root[3])) \
                and (0 == len(self.root[3][self.speech_segment_next_index])):
            self.speech_segment_next_index += 1

    def reset(self):
        self.speech_segment_index = -1
        self.word_index = -1
        self.speech_segment_next_index = 0
        self.word_next_index = 0
        # There are empty speech segments sometimes
        while (self.speech_segment_next_index < len(self.root[3])) \
                and (0 == len(self.root[3][self.speech_segment_next_index])):
            self.speech_segment_next_index += 1

    def read_word(self):
        """
        Read the next word in the transcript file.
        :return: word text content as a string
        """
        logging.debug("\t\tRead word %d %d" % (self.speech_segment_next_index, self.word_next_index))
        if self.is_file_over():
            return ''

        self.speech_segment_index = self.speech_segment_next_index
        self.word_index = self.word_next_index

        self.word_next_index += 1
        # There are empty speech segments sometimes
        while (self.speech_segment_next_index < len(self.root[3])) \
                and (self.word_next_index == len(self.root[3][self.speech_segment_next_index])):
            self.speech_segment_next_index += 1
            self.word_next_index = 0

        word = self.root[3][self.speech_segment_index][self.word_index]
        word_text = word.text.strip()
        return word_text

    def read_speech_segment(self, trim=False):
        """
        Read the next speech segment in the transcript file.
        :param trim: clean the text after concatenating words:
            - delete space after apostrophe (') and dash (-),
            - delete space before fullstop, comma, and dash (.,-)
        :return: speech segment text content as a string
        """
        logging.debug("\tRead speech_segment %d" % self.speech_segment_next_index)
        if self.is_file_over():
            return ''

        words = list()
        speech_segment_current_index = self.speech_segment_next_index

        while self.speech_segment_next_index == speech_segment_current_index:
            word = self.read_word()
            words.append(word)

        speech_segment_text = ' '.join(words)
        # Coup de peigne sur la transcription :
        #   - en supprimant les espaces après les apostrophes et les tirets
        #   - en supprimant les espaces avant un point, une virgule, ou un tiret
        if trim:
            speech_segment_text = re.sub(r"(['-]) ", r"\1", speech_segment_text)
            speech_segment_text = re.sub(r" ([,.-])", r"\1", speech_segment_text)
        return speech_segment_text

    def read_all(self, trim=False):
        """
        Read all the text content of the transcript file.
        :param trim: clean the text after concatenating words:
            - delete space after apostrophe (') and dash (-),
            - delete space before fullstop, comma, and dash (.,-)
        :return: transcript as a string
        """
        logging.debug("Read all %d" % (len(self.root[3]) - self.speech_segment_next_index))
        speech_segments = list()

        while not self.is_file_over():
            speech_segment = self.read_speech_segment(trim=trim)
            speech_segments.append(speech_segment)

        all_text = ' '.join(speech_segments)
        return all_text

    def n_words(self, gender=None):
        """
        Number of words in the transcript.
        :param gender: only account for the given gender (ignore gender if None)
        :return: number of words
        """
        if gender is None:
            n_words = int(self.root[1][0].attrib["nw"])
        else:
            n_words = 0
            for speaker in self.root[2]:
                matching_gender = int(speaker.attrib["gender"]) == gender
                if matching_gender:
                    n_words += int(speaker.attrib["nw"])

        return n_words

    def n_speech_segments(self, no_void=True, gender=None):
        """
        Number of speech segments identified in the audio.
        :param no_void: omit speech segments which contain no words
        :param gender: only account for the given gender (ignore gender if None)
        :return: number of speech segments
        """
        n_speech_segments = 0
        for speech_segment in self.root[3]:
            speaker = self.speakers[speech_segment.attrib["spkid"]]
            matching_gender = int(speaker.attrib["gender"]) == gender
            void_speech_segment = len(speech_segment) == 0
            if (not void_speech_segment or not no_void) and (gender is None or matching_gender):
                n_speech_segments += 1

        return n_speech_segments

    def n_speakers(self, no_void=True, gender=None):
        """
        Number of speakers identified in the audio.
        :param no_void: omit speakers which have void speech duration
        :param gender: only account for the given gender (ignore gender if None)
        :return: number of speakers
        """
        n_speakers = 0
        for speaker in self.root[2]:
            matching_gender = int(speaker.attrib["gender"]) == gender
            # In some ROSETTA transcripts (editor="Vocapia Research"),
            # some speakers would appear in <SpeakerList>, but would actually have no speech time.
            void_speaker = float(speaker.attrib['dur']) == 0.0
            if (not void_speaker or not no_void) and (gender is None or matching_gender):
                n_speakers += 1

        return n_speakers

    def total_duration(self, metadata=True):
        """
        Duration of the audio.
        :param metadata: use the Channel.sigdur metadata in the file
            (else the duration is computed based on the start and on the end of speech segments,
            cropping the potential unspoken time at the beginning or the ending)
        :return: duration in seconds
        """
        if metadata:
            total_duration = float(self.root[1][0].attrib["sigdur"])
        else:
            first_speech_segment_stime = float(self.root[3][0].attrib["stime"])
            last_speech_segment_etime = float(self.root[3][-1].attrib["etime"])
            total_duration = last_speech_segment_etime - first_speech_segment_stime

        return total_duration

    def speech_duration(self, metadata=False, gender=None):
        """
        Duration of the speech.
        :param metadata: use the Channel.spdur|Speaker.dur metadata in the file
            (else the duration is computed by summing all speech segment durations)
        :param gender: only account for the given gender (ignore gender if None)
        :return: duration in seconds
        """
        # In GEM transcripts (editor="LIUM LST"),
        # it seems that Channel.spdur|Speaker.dur correspond to the sum of all word durations,
        # instead of the sum of all speech segment durations
        if metadata:
            # Case where Channel.spdur is returned
            if gender is None:
                speech_duration = float(self.root[1][0].attrib["spdur"])
            # Case where the sum of Speaker.dur (for the matching gender) is returned
            else:
                speech_duration = 0
                for speaker in self.root[2]:
                    matching_gender = int(speaker.attrib["gender"]) == gender
                    if matching_gender:
                        speech_duration += float(speaker.attrib["dur"])
        else:
            speech_duration = 0
            for speech_segment in self.root[3]:
                speaker = self.speakers[speech_segment.attrib["spkid"]]
                matching_gender = int(speaker.attrib["gender"]) == gender
                if gender is None or matching_gender:
                    speech_segment_duration = float(speech_segment.attrib["etime"]) - \
                                              float(speech_segment.attrib["stime"])
                    speech_duration += speech_segment_duration

        return speech_duration

    def word_time_span(self, si, wi):
        word = self.root[3][si][wi]
        stime = float(word.attrib['stime'])
        etime = stime + float(word.attrib['dur'])
        return stime, etime

    def current_word_time_span(self):
        return self.word_time_span(self.speech_segment_index, self.word_index)

    def next_word_time_span(self):
        return self.word_time_span(self.speech_segment_next_index, self.word_next_index)

    def speech_segment_time_span(self, si):
        speech_segment = self.root[3][si]
        stime = float(speech_segment.attrib['stime'])
        etime = float(speech_segment.attrib['etime'])
        return stime, etime

    def current_speech_segment_time_span(self):
        return self.speech_segment_time_span(self.speech_segment_index)

    def next_speech_segment_time_span(self):
        return self.speech_segment_time_span(self.speech_segment_next_index)

    def current_speaker(self):
        speech_segment = self.root[3][self.speech_segment_index]
        spkid = speech_segment.attrib['spkid']
        return spkid

    def current_speaker_gender(self):
        spkid = self.current_speaker()
        speaker = self.speakers[spkid]
        gender = int(speaker.attrib["gender"])
        return gender

    def current_confidence(self):
        word = self.root[3][self.speech_segment_index][self.word_index]
        conf = float(word.attrib['conf'])
        return conf

    def next_word(self):
        if self.is_file_over():
            return ''

        word = self.root[3][self.speech_segment_next_index][self.word_next_index]
        return word.text.strip()

    def is_speech_segment_new(self):
        return self.word_index == 0

    def is_file_over(self):
        return self.speech_segment_next_index >= len(self.root[3])


# FUNCTIONS ############################################################################################################

def read(transcript_file_path, text_file_path, trim=True):
    print('Transcript extraction:')
    print('Initializing...')
    transcript_reader = TranscriptReader(transcript_file_path)
    transcript_reader.n_speakers()
    print('Reading file...')
    transcript_segments = list()
    while not (transcript_reader.is_file_over()):
        transcript_segment = transcript_reader.read_speech_segment(trim=trim)
        transcript_segments.append(transcript_segment)

    print('Writing...')
    utl.write_lines(transcript_segments, text_file_path)


# MAIN  ################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['read'])

    parser.add_argument('--transcript_file', '-traf', type=str)
    parser.add_argument('--text_file', '-txtf', type=str)
    parser.add_argument('--logging_level', '-ll', type=str, choices=cst.LOGGING_LEVELS, default=cst.DEFAULT_LOG_LVL)

    args = parser.parse_args()
    return args


def main(args):
    mode = args.mode

    transcript_file_path = args.transcript_file
    text_file_path = args.text_file

    logging_level = args.logging_level
    logging.basicConfig(format='%(levelname)s:%(message)s', level=getattr(logging, logging_level), force=True)

    if mode == 'read':
        read(transcript_file_path, text_file_path, trim=True)


if __name__ == '__main__':
    main(parse_args())
