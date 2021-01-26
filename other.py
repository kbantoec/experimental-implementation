from __future__ import annotations


def greetings(names: list[str]):
    for name in names:
        print('Hello', name)


if __name__ == '__main__':
    greetings(['Kai', 'Pauline'])
