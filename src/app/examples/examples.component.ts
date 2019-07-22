import { Component, OnInit } from '@angular/core';
import { GithubService } from '../github.service';

@Component({
  selector: 'app-examples',
  templateUrl: './examples.component.html',
  styleUrls: ['./examples.component.css']
})
export class ExamplesComponent implements OnInit {

  examples;

  constructor(private githubService: GithubService) { }

  ngOnInit() {
    this.githubService.getExampleList()
      .subscribe((data) => {
        this.examples = data.filter(e => (e.type == 'dir')).map(e => { return {name: e.name, url: e.html_url}; });
        console.log(this.examples);
      });
  }

}
