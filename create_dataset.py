import click
import requests
import sys
import os


def get_time_data(access_key, path_to_wiki_file, path_to_output_file):
    headers = {"Content-Type": "application/json",
                "Authorization": access_key
                }

    url = "https://www.wikidata.org/w/rest.php/wikibase/v0/entities/items/"

    visited_tuples = []

    with open(path_to_wiki_file, "r") as wiki_file, open(path_to_output_file, "a") as output_file:

        for wline in wiki_file:
            l = wline.split("\t")
            s = l[0]
            r = l[1]
            o = l[2]
            t = l[4]            
            if (s, r) not in visited_tuples:
                visited_tuples.append((s, r))

                response = requests.get(url=url + s + "/statements", headers=headers)

                try:
                    objects = response.json()[r]
                except KeyError:
                    print(f"Fehler: Relation {r} für Entität {s} nicht gefunden")
                    #output_file.write(f"""{s}\t{r}\t{o}\toccurSince\t{t}\toccurUntil\t{t}\n""")
                else:
                    for object in objects:

                        start_time = None
                        end_time = None
                        pit = None

                        try:
                            o = object['value']['content']['amount']
                        except TypeError:
                            try:
                                o = object['value']['content']
                            except KeyError:
                                #output_file.write(f"""{s}\t{r}\t{o}\toccurSince\t{t}\toccurUntil\t{t}\n""")
                                print(f"Fehler: Kein Objekt für Tripel: {s, r, o} gefunden")
                                continue
                        except KeyError:
                            print(f"Fehler: Kein Value für Tripel: {s, r, o} gefunden")
                            continue

                        for i in object['qualifiers']:
                            if i['property']['id'] == 'P580':
                                try:
                                    start_time = i['value']['content']['time'][1:5]
                                except KeyError:
                                    start_time = -1
                            elif i['property']['id'] == 'P582':
                                try:
                                    end_time = i['value']['content']['time'][1:5]
                                except KeyError:
                                    end_time = -1
                            elif i['property']['id'] == 'P585':
                                try:
                                    pit = i['value']['content']['time'][1:5]
                                except KeyError:
                                    pit = -1

                        if pit != None:
                            output_file.write(f"""{s}\t{r}\t{o}\toccurSince\t{pit}\toccurUntil\t{pit}\n""")
                        elif start_time != None or end_time != None:
                            output_file.write(f"""{s}\t{r}\t{o}\toccurSince\t{start_time}\toccurUntil\t{end_time}\n""")

@click.command
@click.option("--access_token", type=str, default="", help="Access token to Wikidata.")
def main(access_token):
    get_time_data(access_token, r"data\original\wiki_train.txt", r"data\raw\train")
    get_time_data(access_token, r"data\original\wiki_test.txt", r"data\raw\test")
    get_time_data(access_token, r"data\original\wiki_valid.txt", r"data\raw\valid")


if __name__ == "__main__":
    main()