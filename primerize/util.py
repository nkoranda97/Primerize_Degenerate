import math
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as pyplot
import numpy
import os
import re
import xlwt

if __package__ is None or not __package__:
    from thermo import calc_Tm
else:
    from .thermo import calc_Tm


class Assembly(object):
    def __init__(self, sequence, primers, name, COL_SIZE=142):
        self.sequence = sequence
        self.primers = primers
        self.name = name
        (self.bp_lines, self.seq_lines, self.print_lines, self.Tm_overlaps) = draw_assembly(self.sequence, self.primers, COL_SIZE)

    def __repr__(self):
        return '\033[94m%s\033[0m {\n    \033[93m\'primers\'\033[0m: %s, \n    \033[93m\'seq_lines\'\033[0m: \033[91mlist\033[0m(\033[91mstring\033[0m * %d), \n    \033[93m\'bp_lines\'\033[0m: \033[91mlist\033[0m(\033[91mstring\033[0m * %d), \n    \033[93m\'print_lines\'\033[0m: \033[91mlist\033[0m(\033[91mtuple\033[0m * %d), \n    \033[93m\'Tm_overlaps\'\033[0m: %s\n}' % (self.__class__, repr(self.primers), len(self.seq_lines), len(self.bp_lines), len(self.print_lines), repr(self.Tm_overlaps))

    def __str__(self):
        return self.echo()


    def echo(self):
        output = ''
        x = 0
        for i in range(len(self.print_lines)):
            (flag, string) = self.print_lines[i]
            if (flag == '$' and 'xx' in string):
                Tm = '%2.1f' % self.Tm_overlaps[x]
                output += string.replace('x' * len(Tm), '\033[41m%s\033[0m' % Tm) + '\n'
                x += 1
            elif (flag == '^' or flag == '!'):
                num = ''.join(re.findall("[0-9]+", string))
                string = string.replace(num, '\033[100m%s\033[0m' % num) + '\n'
                if flag == '^':
                    string = string.replace('A', '\033[94mA\033[0m').replace('G', '\033[94mG\033[0m').replace('C', '\033[94mC\033[0m').replace('T', '\033[94mT\033[0m')
                else:
                    string = string.replace('A', '\033[95mA\033[0m').replace('G', '\033[95mG\033[0m').replace('C', '\033[95mC\033[0m').replace('T', '\033[95mT\033[0m')
                output += string
            elif (flag == '~'):
                output += '\033[92m%s\033[0m' % string + '\n'
            elif (flag == '='):
                output += '\033[96m%s\033[0m' % string + '\n'
            else:
                output += string + '\n'
        return output[:-1]


    def save(self, path='./', name=None):
        if name is None: name = self.name
        f = open(os.path.join(path, '%s_assembly.txt' % name), 'w')
        lines = self.echo().replace('\033[0m', '').replace('\033[100m', '').replace('\033[92m', '').replace('\033[93m', '').replace('\033[94m', '').replace('\033[95m', '').replace('\033[96m', '').replace('\033[41m', '')
        f.write(lines)
        f.close()



class Plate_96Well(object):
    def __init__(self):
        self.coords = set()
        self._data = {}

    def __repr__(self):
        if len(self):
            return '\033[94m%s\033[0m {\033[93m\'coords\'\033[0m: %s, \033[93m\'data\'\033[0m: \033[91mdict\033[0m(\033[91mtuple\033[0m * %d)}' % (self.__class__, ' '.join(sorted(self.coords)), len(self._data))
        else:
            return '\033[94m%s\033[0m (empty)' % self.__class__

    def __str__(self):
        return self.echo()

    def __len__(self):
        return len(self.coords)


    def has(self, coord):
        return coord in self.coords


    def get(self, coord):
        if coord.lower() == 'count':
            return len(self.coords)
        else:
            coord = format_coord(coord)
            if coord_to_num(coord) == -1:
                raise AttributeError('\033[41mERROR\033[0m: Illegal coordinate value \033[95m%s\033[0m for \033[94m%s.get()\033[0m.\n' % (coord, self.__class__))
            elif coord in self.coords:
                return self._data[coord_to_num(coord)]
            else:
                raise KeyError('\033[41mERROR\033[0m: Non-Existent coordinate value \033[95m%s\033[0m for \033[94m%s.get()\033[0m.\n' % (coord, self.__class__))


    def set(self, coord, tag, primer):
        coord = format_coord(coord)
        if coord_to_num(coord) == -1:
            raise AttributeError('\033[41mERROR\033[0m: Illegal coordinate value \033[95m%s\033[0m for \033[94m%s.set()\033[0m.\n' % (coord, self.__class__))
        else:
            self.coords.add(coord)
            self._data[coord_to_num(coord)] = (tag, primer)


    def reset(self):
        self.coords = set()
        self._data = {}


    def echo(self, ref_primer=''):
        return print_primer_plate(self, ref_primer)


    def save(self, file_name='./', title=''):
        fig = pyplot.figure()
        pyplot.axes().set_aspect('equal')
        pyplot.axis([0, 13.875, 0, 9.375])
        pyplot.xticks([x * 1.125 + 0.75 for x in range(12)], [str(x + 1) for x in range(12)], fontsize=14)
        pyplot.yticks([y * 1.125 + 0.75 for y in range(8)], list('ABCDEFGH'), fontsize=14)
        pyplot.suptitle(title, fontsize=16, fontweight='bold')
        ax = pyplot.gca()
        for edge in ('bottom', 'top', 'left', 'right'):
            ax.spines[edge].set_color('w')
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False

        (x_green, x_violet, x_gray, y_green, y_violet, y_gray) = ([], [], [], [], [], [])
        for i in range(8):
            for j in range(12):
                num = i + j * 8 + 1
                if num_to_coord(num) in self.coords:
                    if 'WT' in self._data[num][0]:
                        x_green.append(j * 1.125 + 0.75)
                        y_green.append(i * 1.125 + 0.75)
                    else:
                        x_violet.append(j * 1.125 + 0.75)
                        y_violet.append(i * 1.125 + 0.75)
                else:
                    x_gray.append(j * 1.125 + 0.75)
                    y_gray.append(i * 1.125 + 0.75)
        pyplot.scatter(x_gray, y_gray, 961, c='#ffffff', edgecolor='#333333', linewidth=5)
        pyplot.scatter(x_violet, y_violet, 961, c='#ecddf4', edgecolor='#c28fdd', linewidth=5)
        pyplot.scatter(x_green, y_green, 961, c='#beebde', edgecolor='#29be92', linewidth=5)

        matplotlib.rcParams['svg.fonttype'] = 'none'
        pyplot.savefig(file_name, orientation='landscape', format='svg')



class Mutation(object):
    def __init__(self, mut_str=[]):
        self._data = {}
        if mut_str: self.push(mut_str)

    def __repr__(self):
        return '\033[94m%s\033[0m' % self.__class__

    def __str__(self):
        return self.echo()

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, Mutation): other = other.list()
        return self.has(other)


    def has(self, mut_str):
        if not isinstance(mut_str, list): mut_str = [mut_str]
        if not (mut_str or self._data): return True
        flag = False

        for mut in mut_str:
            seq_org = mut[0]
            seq_mut = mut[-1]
            seq_pos = int(mut[1:-1])

            if seq_pos in self._data and self._data[seq_pos] == (seq_org, seq_mut): flag = True

        return flag


    def push(self, mut_str):
        if not isinstance(mut_str, list): mut_str = [mut_str]
        for mut in mut_str:
            if mut == 'WT': continue

            seq_org = mut[0]
            seq_mut = mut[-1]
            seq_pos = int(mut[1:-1])
            self._data[seq_pos] = (seq_org, seq_mut)


    def pop(self, mut_str):
        if not isinstance(mut_str, list): mut_str = [mut_str]
        for mut in mut_str:
            if self.has(mut):
                seq_pos = int(mut[1:-1])
                self._data.pop(seq_pos, None)
            else:
                return False

        return True


    def list(self):
        mut_list = []
        for seq_pos in self._data:
            mut_list.append('%s%d%s' % (self._data[seq_pos][0], seq_pos, self._data[seq_pos][1]))
        return mut_list


    def echo(self):
        output = ', '.join(self.list())
        if not output: output = 'WT'
        return output



class Construct_List(object):
    def __init__(self):
        self._data = []

    def __repr__(self):
        if len(self):
            return '\033[94m%s\033[0m {\033[91mlist\033[0m(%s * %d)}' % (self.__class__, repr(Mutation()), len(self))
        else:
            return '\033[94m%s\033[0m (empty)' % self.__class__

    def __str__(self):
        return self.echo()

    def __len__(self):
        return len(self._data)


    def has(self, mut_list):
        if not isinstance(mut_list, Mutation): mut_list = Mutation(mut_list)
        for construct in self._data:
            if construct == mut_list: return True
        return False


    def push(self, mut_list):
        if not isinstance(mut_list, Mutation): mut_list = Mutation(mut_list)
        if self.has(mut_list): return
        self._data.append(mut_list)


    def pop(self, mut_list):
        if not isinstance(mut_list, Mutation): mut_list = Mutation(mut_list)
        for i, construct in enumerate(self._data):
            if construct == mut_list:
                self._data.pop(i)
                return True
        return False


    def echo(self, prefix=''):
        output = ''
        for construct in self._data:
            output += prefix + construct.echo() + '\n'
        return output



def DNA2RNA(sequence):
    return sequence.upper().replace('T', 'U')


def RNA2DNA(sequence):
    return sequence.upper().replace('U', 'T')


def complement(sequence):
    sequence = list(sequence)
    rc_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    for i in range(len(sequence)):
        sequence[i] = rc_dict[sequence[i]]
    return ''.join(sequence)


def reverse_complement(sequence):
    return complement(sequence[::-1])


def primer_suffix(num):
    if num % 2:
        return '\033[95m R\033[0m'
    else:
        return '\033[94m F\033[0m'


def draw_assembly(sequence, primers, COL_SIZE):
    N_primers = primers.shape[1]
    seq_line_prev = list(' ' * max(len(sequence), COL_SIZE))
    bp_lines = []
    seq_lines = []
    Tms = []

    for i in range(N_primers):
        primer = primers[:, i]
        seq_line = list(' ' * max(len(sequence), COL_SIZE))
        (seg_start, seg_end, seg_dir) = primer

        if (seg_dir == 1):
            for j in range(seg_start, seg_end + 1):
                seq_line[j] = sequence[j]

            if (seg_end + 1 < len(sequence)):
                seq_line[seg_end + 1] = '-'
            if (seg_end + 2 < len(sequence)):
                seq_line[seg_end + 2] = '>'

            num_txt = '%d' % (i + 1)
            if (seg_end + 4 + len(num_txt) < len(sequence)):
                offset = seg_end + 4
                seq_line[offset:(offset + len(num_txt))] = num_txt
        else:
            for j in range(seg_start, seg_end + 1):
                seq_line[j] = reverse_complement(sequence[j])

            if (seg_start - 1 >= 0):
                seq_line[seg_start - 1] = '-'
            if (seg_start - 2 >= 0):
                seq_line[seg_start - 2] = '<'

            num_txt = '%d' % (i + 1)
            if (seg_start - 3 - len(num_txt) >= 0):
                offset = seg_start - 3 - len(num_txt)
                seq_line[offset:(offset + len(num_txt))] = num_txt

        bp_line = list(' ' * max(len(sequence), COL_SIZE))
        overlap_seq = ''
        last_bp_pos = 1
        for j in range(len(sequence)):
            if (seq_line_prev[j] in 'ACGT' and seq_line[j] in 'ACGT'):
                bp_line[j] = '|'
                last_bp_pos = j
                overlap_seq += sequence[j]

        if (last_bp_pos > 1):
            Tm = calc_Tm(overlap_seq, 0.2e-6, 0.1, 0.0015)
            Tms.append(Tm)
            Tm_txt = '%2.1f' % Tm
            offset = last_bp_pos + 2
            bp_line[offset:(offset + len(Tm_txt))] = 'x' * len(Tm_txt)

        bp_lines.append(''.join(bp_line))
        seq_lines.append(''.join(seq_line))
        seq_line_prev = seq_line

    print_lines = []
    for i in range(int(math.floor((len(sequence) - 1) / COL_SIZE)) + 1):
        start_pos = COL_SIZE * i
        end_pos = min(COL_SIZE * (i + 1), len(sequence))
        out_line = sequence[start_pos:end_pos]
        print_lines.append(('~', out_line))

        for j in range(len(seq_lines)):
            if (len(bp_lines[j][end_pos:].replace(' ', '')) and ('|' not in bp_lines[j][end_pos:].replace(' ', '')) and (not len(bp_lines[j][:start_pos].replace(' ', '')))):
                bp_line = bp_lines[j][start_pos:].rstrip()
            elif ('|' not in bp_lines[j][start_pos:end_pos]):
                bp_line = ' ' * (end_pos - start_pos + 1)
            else:
                bp_line = bp_lines[j][start_pos:end_pos]
            seq_line = seq_lines[j][start_pos:end_pos]

            if len(bp_line.replace(' ', '')) or len(seq_line.replace(' ', '')):
                print_lines.append(('$', bp_line))
                print_lines.append(('^!'[j % 2], seq_line))
        print_lines.append(('$', ' ' * (end_pos - start_pos + 1)))
        print_lines.append(('=', complement(out_line)))
        print_lines.append(('', '\n'))

    return (bp_lines, seq_lines, print_lines, Tms)


def format_coord(coord):
    return coord[0] + coord[1:].zfill(2)


def coord_to_num(coord):
    coord = coord.upper().strip()
    row = 'ABCDEFGH'.find(coord[0])
    if row == -1: return -1
    col = int(coord[1:])
    if col < 0 or col > 12: return -1
    return (col - 1) * 8 + row + 1


def num_to_coord(num):
    if num < 0 or num > 96: return -1
    row = 'ABCDEFGH'[(num - 1) % 8]
    col = (num - 1) / 8 + 1
    return '%s%0*d' % (row, 2, col)


def get_mut_range(mut_start, mut_end, offset, sequence):
    if (not mut_start) or (mut_start is None): mut_start = 1 - offset
    mut_start = min(max(mut_start, 1 - offset), len(sequence) - offset)
    if (not mut_end) or (mut_end is None): mut_end = len(sequence) - offset
    mut_end = max(min(mut_end, len(sequence) - offset), 1 - offset)
    which_muts = list(range(mut_start, mut_end + 1))
    return (which_muts, mut_start, mut_end)


def get_primer_index(primer_set, sequence):
    N_primers = len(primer_set)
    coverage = numpy.zeros((1, len(sequence)))
    primers = numpy.zeros((3, N_primers))

    for n in range(N_primers):
        primer = RNA2DNA(primer_set[n])
        if n % 2:
            i = sequence.find(reverse_complement(primer))
        else:
            i = sequence.find(primer)
        if i == -1:
            return ([], False)
        else:
            start_pos = i
            end_pos = i + len(primer_set[n]) - 1
            seq_dir = math.copysign(1, 0.5 - n % 2)
            primers[:, n] = [start_pos, end_pos, seq_dir]
            coverage[0, start_pos:(end_pos + 1)] = 1

    return (primers.astype(int), coverage.all())


def get_mutation(nt, lib):
    idx = 'ATCG'.find(nt)
    if lib == 1:
        return 'TAGC'[idx]
    elif lib == 2:
        return 'CCAA'[idx]
    elif lib == 3:
        return 'GGTT'[idx]


def print_primer_plate(plate, ref_primer):
    string = ''
    for key in sorted(plate._data):
        string += '\033[94m%s\033[0m' % num_to_coord(key).ljust(5)
        mut = plate._data[key][0]
        if mut[-2:] == 'WT':
            string += ('%s\033[100m%s\033[0m' % (mut[:-2], mut[-2:])).ljust(28)
        else:
            string += ('%s\033[96m%s\033[0m\033[93m%s\033[0m\033[91m%s\033[0m' % (mut[:5], mut[5], mut[6:-1], mut[-1])).ljust(45)

        if ref_primer:
            for i in range(len(ref_primer)):
                if ref_primer[i] != plate._data[key][1][i]:
                    string += '\033[41m%s\033[0m' % plate._data[key][1][i]
                else:
                    string += plate._data[key][1][i]
        else:
            string += plate._data[key][1]
        string += '\n'

    if not string: string = '(empty)\n'
    return string


def save_plate_layout(plates, N_plates, N_primers, prefix, path):
    for k in range(N_plates):
        for p in range(N_primers):
            primer_sequences = plates[p][k]
            num_primers_on_plate = primer_sequences.get('count')

            if num_primers_on_plate:
                if num_primers_on_plate == 1 and primer_sequences.has('A01') and 'WT' in primer_sequences.get('A01')[0]: continue

                file_name = os.path.join(path, '%s_plate_%d_primer_%d.svg' % (prefix, k + 1, p + 1))
                print('Creating plate image: \033[94m%s\033[0m.' % file_name)
                title = '%s_plate_%d_primer_%d' % (prefix, k + 1, p + 1)
                primer_sequences.save(file_name, title)


def save_construct_key(keys, name, path, prefix=''):
    prefix = 'Lib%s-' % prefix if prefix else ''
    f = open(os.path.join(path, '%s_keys.txt' % name), 'w')
    print('Creating keys file ...')
    f.write(keys.echo(prefix))
    f.close()


def save_plates_excel(plates, N_plates, N_primers, prefix, path):
    for k in range(N_plates):
        file_name = os.path.join(path, '%s_plate_%d.xls' % (prefix, k + 1))
        print('Creating plate file: \033[94m%s\033[0m.' % file_name)
        workbook = xlwt.Workbook()

        for p in range(N_primers):
            primer_sequences = plates[p][k]
            num_primers_on_plate = primer_sequences.get('count')

            if num_primers_on_plate:
                if num_primers_on_plate == 1 and primer_sequences.has('A01') and 'WT' in primer_sequences.get('A01')[0]: continue

                sheet = workbook.add_sheet('primer_%d' % (p + 1))
                sheet.col(1).width = 256 * 15
                sheet.col(2).width = 256 * 75

                sheet.write(0, 0, 'WellPosition', xlwt.easyxf('font: bold 1'))
                sheet.write(0, 1, 'Name', xlwt.easyxf('font: bold 1'))
                sheet.write(0, 2, 'Sequence', xlwt.easyxf('font: bold 1'))
                sheet.write(0, 3, 'Notes', xlwt.easyxf('font: bold 1'))

                for i, row in enumerate(sorted(primer_sequences._data)):
                    if 'WT' in primer_sequences._data[row][0]:
                        sheet.write(i + 1, 0, num_to_coord(row), xlwt.easyxf('font: color blue, italic 1'))
                        sheet.write(i + 1, 1, primer_sequences._data[row][0], xlwt.easyxf('font: color blue'))
                        sheet.write(i + 1, 2, primer_sequences._data[row][1], xlwt.easyxf('font: color blue'))
                    else:
                        sheet.write(i + 1, 0, num_to_coord(row), xlwt.easyxf('font: italic 1'))
                        sheet.write(i + 1, 1, primer_sequences._data[row][0])
                        sheet.write(i + 1, 2, primer_sequences._data[row][1])

        workbook.save(file_name)


def draw_region(sequence, params):
    offset = params['offset']
    start = params['which_muts'][0] + offset - 1
    end = params['which_muts'][-1] + offset - 1
    fragments = []

    if start <= 20:
        fragments.append(sequence[:start])
    else:
        fragments.append(sequence[:10] + '......' + sequence[start - 10:start])
    if end - start <= 40:
        fragments.append(sequence[start:end + 1])
    else:
        fragments.append(sequence[start:start + 20] + '......' + sequence[end - 19:end + 1])
    if len(sequence) - end <= 20:
        fragments.append(sequence[end + 1:])
    else:
        fragments.append(sequence[end + 1:end + 11] + '......' + sequence[-10:])

    labels = ['%d' % (1 - offset), '%d' % params['which_muts'][0], '%d' % params['which_muts'][-1], '%d' % (len(sequence) - offset)]
    (illustration_1, illustration_2, illustration_3) = ('', '', '')

    if len(fragments[0]) >= len(labels[0]):
        illustration_1 += '\033[91m' + fragments[0][0] + '\033[0m\033[40m' + fragments[0][1:] + '\033[0m'
        illustration_2 += '\033[91m|%s\033[0m' % (' ' * (len(fragments[0]) - 1))
        illustration_3 += '\033[91m%s%s\033[0m' % (labels[0], ' ' * (len(fragments[0]) - len(labels[0])))
    elif fragments[0]:
        illustration_1 += '\033[91m' + fragments[0][0] + '\033[0m\033[40m' + fragments[0][1:] + '\033[0m'
        illustration_2 += '\033[91m|%s\033[0m' % (' ' * len(fragments[0]))
        illustration_3 += '\033[91m|%s\033[0m' % (' ' * len(fragments[0]))

    if len(fragments[1]) >= len(labels[1]) + len(labels[2]):
        illustration_1 += '\033[44m' + fragments[1][0] + '\033[0m\033[46m' + fragments[1][1:-1] + '\033[0m\033[44m' + fragments[1][-1] + '\033[0m'
        illustration_2 += '\033[92m|%s|\033[0m' % (' ' * (len(fragments[1]) - 2))
        illustration_3 += '\033[92m%s%s%s\033[0m' % (labels[1], ' ' * (len(fragments[1]) - len(labels[1]) - len(labels[2])), labels[2])
    elif fragments[1]:
        if len(fragments[1]) >= len(labels[1]):
            illustration_1 += '\033[44m' + fragments[1][0] + '\033[0m\033[46m' + fragments[1][1:] + '\033[0m'
            illustration_2 += '\033[92m|%s\033[0m' % (' ' * (len(fragments[1]) - 1))
            illustration_3 += '\033[92m%s%s\033[0m' % (labels[1], ' ' * (len(fragments[1]) - len(labels[1])))
        else:
            illustration_1 += '\033[46m' + fragments[1] + '\033[0m'
            illustration_2 += '\033[92m|%s\033[0m' % (' ' * len(fragments[1]))
            illustration_3 += '\033[92m|%s\033[0m' % (' ' * len(fragments[1]))

    if len(fragments[2]) >= len(labels[3]):
        illustration_1 += '\033[40m' + fragments[2][:-1] + '\033[0m\033[91m' + fragments[2][-1] + '\033[0m'
        illustration_2 += '\033[91m%s|\033[0m' % (' ' * (len(fragments[2]) - 1))
        illustration_3 += '\033[91m%s%s\033[0m' % (' ' * (len(fragments[2]) - len(labels[3])), labels[3])
    elif fragments[2]:
        illustration_1 += '\033[40m' + fragments[2][:-1] + '\033[0m\033[91m' + fragments[2][-1] + '\033[0m'
        illustration_2 += '\033[91m|%s\033[0m' % (' ' * len(fragments[2]))
        illustration_3 += '\033[91m|%s\033[0m' % (' ' * len(fragments[2]))

    return {'labels': labels, 'fragments': fragments, 'lines': (illustration_1, illustration_2, illustration_3)}

