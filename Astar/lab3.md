# Ćwiczenie

W bieżącym katalogu jest mały projekt zawierający przykładową implementację algorytmu A*. Jest to w 2 wersjach - uproszczona, aby było widać główną część kodu i minimum narzutu (```astar_mini.py```), oraz rozszerzona, która zapisuje część kroków w plikach png do dalszego przeglądu (```astar_test.py```). Implementacja jest uniwersalna i nie zakłada ona żadnego typu danych przypisanego do reprezentacji grafu. Graf jest obsługiwany za pomocą funkcji ```accessible``` która zwraca węzły sąsiadujące z podanym w argumencie. Ta funkcja musi zwracać tablicę zawierającą elementy dokładnie takiego samego typu jak argument (węzeł który będzie sprawdzany pod względem sąsiadów). W załączonej implementacji A*, funkcja heurystyczna i funkcja odległości jest taka sama i nazywa się ```h_function```. Przykład wykorzystuje dodatkową bibliotekę do obsługi obrazów - ```Pillow```. Wersja ```astar_mini.py``` nie wymaga żadnych dodatkowych zależności.

Zadanie oddajemy w formie plików źródłowych oraz odpowiedzi na pytania w formie PDF/MD/TXT/...

## 0. Decyzja

Zdecyduj, czy chcesz wykonać pełną wersję ćwiczenia z opcją na bonus, czy też nie. Jeśli wersja z możliwym bonusem, to proponuję od razu wersję ```astar_test.py```.

## 1. Kompilacja

Proszę załaduj projekt do wybranego środowiska programistycznego (IDE) i skompiluj. Gotowy program będzie ładował plik ```img.png``` z bieżącego katalogu i zapisywał wynik w pliku ```result.png```. Przypomnij sobie temat ścieżek relatywnych, absolutnych oraz dowiedz się, gdzie uruchamiany jest Twój projekt.

## 2. Analiza

Dlaczego trasy są generowane po prostokątach? Podpowiem, że należy przyjrzeć się funkcji heurystycznej i funkcji ```accessible```. Uzasadnij odpowiedź, a jeśli nie jesteś w stanie nic wymyślić, poproś prowadzącego.

## 3. Różne wersje

Przygotuj 3 wersje funkcji heurystycznej. Mają realizować następujące metryki (jedna już jest):

* Metryka Manhattan
* Metryka Euklidesowa
* Metryka losowa (skorzystaj z biblioteki STL, a konkretniej random oraz random_device do zainicjowania)

Niech to będą oddzielne funkcje w kodzie. Staraj się aby miały spójną konwencję nazewnictwa oraz zachowaj standardy dobrego kodu. Opisz napisane funkcje w standardzie Doxygen.

## 4. Parametryzacja

Niech program pozwala na wybór metryki za pomocą linii komend. Uzytkownik powinien mieć możliwość wyboru bez konieczności rekompilacji.

## 5. Testowanie

### 5.1 Długość trasy

Dodaj do kodu wypisywanie długości trasy (czyli informacja o liczbie węzłów ścieżki zwracanej). Porównaj jakie wyniki daje użycie poszczególnych funkcji heurystycznych.

### 5.2 Czas obliczeń

Umożliw pomiar czasu obliczeń i porównaj czasy wykonania dla każdej metryki. Jaka kombinacja daje najszybszy wynik, oraz jaka jest Twoja interpretacja (czyli dlaczego tak wychodzi)?

## 6. Zadanie domowe 

### 6.1 UKOS

Rozwiń funkcję generującą sąsiedztwo o możliwość przejścia na ukos (czyli jednocześnie po x i y). Przetestuj metryki które zostały stworzone wcześniej i zobacz jak teraz tworzą się ścieżki.


### 6.2 Porównanie czasów 

Porównaj czas wykonania programu w Pythonie i w C++. Z czego wynika różnica? Zinterpretuj wyniki.

## 7. Zadanie dodatkowe

Rozwiń ten przykład, o sytuację, gdzie teren ma różną trudność, to znaczy, po niektórych obszarach koszt przejścia będzie większy, a po niektórych będzie mniejszy. Można to przedstawić na grafice ```.png``` za pomocą odcieni.
